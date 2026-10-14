# ============================================================
# Y-Net for 12-Class Martian Terrain Segmentation
# EfficientNet-B0 + AGCB + CSAIB + FARB + Boundary Branch
# Rare-class-aware sampling + Weighted CE + Class-aware Dice
# + Focal + Tversky + Boundary loss + full evaluation
# ============================================================

import os
import re
import math
import random
import zipfile
from pathlib import Path

import time
import gc
import psutil
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

# -----------------------------
# 0. Configuration
# -----------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "data.zip"
EXTRACT_DIR = BASE_DIR / "data"
LABEL_DIR = EXTRACT_DIR / "label_masks"

NUM_CLASSES = 12
IMG_SIZE = 384
# Keep 384 for stable Colab GPU memory; use 512 only if VRAM allows.
BATCH_SIZE = 2
NUM_WORKERS = 0
EPOCHS = 120
LR = 2e-4
ENCODER_LR = 5e-5
WARMUP_EPOCHS = 8
WEIGHT_DECAY = 1e-4
PATIENCE = 25
AMP = True
GRAD_ACCUM_STEPS = 2
USE_TTA = True

VAL_RATIO = 0.20
USE_PRETRAINED = True
USE_RARE_CLASS_SAMPLER = True
RARE_PIXEL_THRESHOLD = 0.025
MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE = 500

CHECKPOINT_PATH = "best_y_net_mars.pth"
PRED_DIR = "y_net_predictions"

# These are the 12 RGB colors found in your uploaded masks.
# The numerical class IDs are assigned in this order.
# Replace CLASS_NAMES with your actual terrain names if you have
# the dataset's semantic legend.
PALETTE_RGB = [
    (0, 0, 0),
    (165, 42, 42),
    (128, 128, 0),
    (255, 192, 203),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 165, 0),
    (0, 255, 255),
    (128, 0, 128),
    (0, 0, 255),
    (255, 0, 0),
]

CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------
# 1. Reproducibility
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type != "cuda":
    print("WARNING: CUDA GPU not detected. Training will run on CPU.")


# -----------------------------
# 2. Extract dataset
# -----------------------------
def find_zip():
    if os.path.exists(ZIP_PATH):
        return ZIP_PATH

    try:
        from google.colab import files
        print("ZIP not found at:", ZIP_PATH)
        print("Upload train_aug_combined_final.zip")
        uploaded = files.upload()
        for name in uploaded:
            if name.lower().endswith(".zip"):
                return os.path.join("/content", name)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Dataset ZIP not found. Set ZIP_PATH correctly: {ZIP_PATH}"
    )

zip_file = find_zip()

if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(EXTRACT_DIR)

train_image_dir = EXTRACT_DIR / "train" / "images"
train_mask_dir = EXTRACT_DIR / "train" / "masks"

if train_image_dir.is_dir() and train_mask_dir.is_dir():
    root_candidates = [train_image_dir]
    mask_candidates = [train_mask_dir]
else:
    root_candidates = list(Path(EXTRACT_DIR).rglob("images"))
    mask_candidates = list(Path(EXTRACT_DIR).rglob("masks"))

if not root_candidates or not mask_candidates:
    raise RuntimeError(
        "Could not find images/ and masks/ directories after extraction."
    )

IMAGE_DIR = root_candidates[0]
MASK_DIR = mask_candidates[0]

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

print("Image directory:", IMAGE_DIR)
print("Mask directory :", MASK_DIR)


# -----------------------------
# 3. Pair images and masks
# -----------------------------
def build_pairs(image_dir, mask_dir):
    image_files = sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    mask_files = sorted(Path(mask_dir).glob("*.png"))

    mask_by_stem = {mask.stem: mask for mask in mask_files}

    pairs = []
    for image in image_files:
        mask = mask_by_stem.get(image.stem)
        if mask is not None:
            pairs.append((image, mask))

    return pairs


pairs = build_pairs(IMAGE_DIR, MASK_DIR)

if len(pairs) == 0:
    raise RuntimeError("No image-mask pairs found.")

print(f"Found {len(pairs)} image-mask pairs.")

for im_path, mask_path in pairs[:5]:
    im = Image.open(im_path)
    mask = Image.open(mask_path)
    print(
        im_path.name, im.size, im.mode,
        "|",
        mask_path.name, mask.size, mask.mode
    )


# -----------------------------
# 4. Convert RGB masks to class-ID masks
# -----------------------------
palette = np.asarray(PALETTE_RGB, dtype=np.int16)


def rgb_mask_to_labels(mask_rgb):
    """
    Convert RGB segmentation mask to class IDs.

    Exact palette pixels are mapped to their class.
    Anti-aliased/blended pixels introduced by augmentation/resizing
    are mapped to the nearest of the 12 known class colors.
    """
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)

    labels = np.empty(flat.shape[0], dtype=np.uint8)

    chunk = 250_000
    for start in range(0, len(flat), chunk):
        end = min(start + chunk, len(flat))
        x = flat[start:end].astype(np.int32)

        distances = (
            (x[:, None, :] - palette[None, :, :]) ** 2
        ).sum(axis=2)

        labels[start:end] = np.argmin(
            distances, axis=1
        ).astype(np.uint8)

    return labels.reshape(h, w)


def convert_masks_to_labels(pairs):
    converted = []

    for idx, (im_path, mask_path) in enumerate(pairs):
        out_path = Path(LABEL_DIR) / f"{mask_path.stem}.png"

        if not out_path.exists():
            rgb = Image.open(mask_path).convert("RGB")
            labels = rgb_mask_to_labels(rgb)
            Image.fromarray(labels, mode="L").save(out_path)

        converted.append((im_path, out_path))

        if (idx + 1) % 100 == 0 or idx == 0:
            print(
                f"Converted masks: {idx + 1}/{len(pairs)}"
            )

    return converted


pairs = convert_masks_to_labels(pairs)


# -----------------------------
# 5. Dataset statistics
# -----------------------------
def mask_stats(mask_path):
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    return np.bincount(
        mask.reshape(-1),
        minlength=NUM_CLASSES
    )


all_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
image_class_counts = []

for _, mask_path in pairs:
    counts = mask_stats(mask_path)
    all_counts += counts
    image_class_counts.append(counts)

total_pixels = all_counts.sum()
freq = all_counts / max(total_pixels, 1)

print("\n========== DATASET CLASS DISTRIBUTION ==========")

for c in range(NUM_CLASSES):
    image_presence = sum(
        counts[c] > 0
        for counts in image_class_counts
    )

    print(
        f"{c:2d} | {CLASS_NAMES[c]:10s} | "
        f"pixels={all_counts[c]:12d} | "
        f"{freq[c]*100:7.3f}% | "
        f"images={image_presence:4d}"
    )

rare_classes = np.where(
    freq < RARE_PIXEL_THRESHOLD
)[0].tolist()

print("Rare classes:", rare_classes)


# -----------------------------
# 6. Train / validation split
# -----------------------------
# IMPORTANT:
# The ZIP contains both original and augmented images.
# If the augmented images were generated from the same originals,
# a random image-level split can cause augmentation leakage.
#
# The ZIP does not contain an original->augmentation mapping file,
# so this script uses a reproducible split stratified toward images
# containing rare classes.
#
# For a publication-grade experiment, use a group split if you have
# the original/augmentation mapping.

rng = np.random.default_rng(SEED)

indices = np.arange(len(pairs))

rare_presence = np.array([
    any(
        counts[c] >= MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE
        for c in rare_classes
    )
    if rare_classes else False
    for counts in image_class_counts
])

rare_idx = indices[rare_presence]
normal_idx = indices[~rare_presence]

rng.shuffle(rare_idx)
rng.shuffle(normal_idx)


def split_group(idx_array, val_ratio):
    n_val = max(
        1,
        int(round(len(idx_array) * val_ratio))
    )
    return idx_array[n_val:], idx_array[:n_val]


train_r, val_r = split_group(rare_idx, VAL_RATIO)
train_n, val_n = split_group(normal_idx, VAL_RATIO)

train_indices = np.concatenate([train_r, train_n])
val_indices = np.concatenate([val_r, val_n])

rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_pairs = [pairs[i] for i in train_indices]
val_pairs = [pairs[i] for i in val_indices]

print("\nTrain pairs:", len(train_pairs))
print("Val pairs  :", len(val_pairs))


# -----------------------------
# 7. Training class weights
# -----------------------------
train_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
train_image_counts = []

for _, mask_path in train_pairs:
    counts = mask_stats(mask_path)
    train_counts += counts
    train_image_counts.append(counts)

train_freq = train_counts / max(
    train_counts.sum(), 1
)

# Smoothed inverse-frequency weighting.
class_weights = 1.0 / np.sqrt(
    np.maximum(train_freq, 1e-8)
)

class_weights /= class_weights.mean()
class_weights = np.clip(
    class_weights,
    0.25,
    4.0
)

print("\nClass weights:")

for c, w in enumerate(class_weights):
    print(
        f"{c:2d} {CLASS_NAMES[c]:10s}: {w:.3f}"
    )


# -----------------------------
# 8. Rare-class-aware sampler
# -----------------------------
def make_sampler_weights(image_counts):
    weights = []

    for counts in image_counts:
        score = 1.0

        for c in rare_classes:
            if (
                counts[c]
                >= MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE
            ):
                score += 1.5 * float(class_weights[c])

        weights.append(score)

    return torch.as_tensor(
        weights,
        dtype=torch.double
    )


sampler = None
shuffle = True

if USE_RARE_CLASS_SAMPLER:
    sampler_weights = make_sampler_weights(
        train_image_counts
    )

    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True,
    )

    shuffle = False


# -----------------------------
# 9. Paired augmentations
# -----------------------------
def paired_train_transform(image, mask):
    w, h = image.size

    # Random scale + crop gives the network substantially more
    # variation than flips/rotations alone while keeping image/mask aligned.
    scale = random.uniform(0.72, 1.0)
    crop_w = max(1, min(w, int(w * scale)))
    crop_h = max(1, min(h, int(h * scale)))

    if crop_w < IMG_SIZE or crop_h < IMG_SIZE:
        # Pad/crop is handled by resizing below.
        left = random.randint(0, max(0, w - crop_w))
        top = random.randint(0, max(0, h - crop_h))
    else:
        left = random.randint(0, max(0, w - crop_w))
        top = random.randint(0, max(0, h - crop_h))

    image = TF.crop(image, top, left, crop_h, crop_w)
    mask = TF.crop(mask, top, left, crop_h, crop_w)

    image = TF.resize(
        image, [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.BILINEAR
    )
    mask = TF.resize(
        mask, [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.NEAREST
    )

    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    if random.random() < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    if random.random() < 0.5:
        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            image = TF.rotate(
                image, angle,
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.rotate(
                mask, angle,
                interpolation=InterpolationMode.NEAREST
            )

    # Mild photometric augmentation; do not over-distort Mars textures.
    if random.random() < 0.7:
        image = TF.adjust_brightness(
            image, random.uniform(0.88, 1.12)
        )
        image = TF.adjust_contrast(
            image, random.uniform(0.88, 1.15)
        )

    if random.random() < 0.15:
        image = TF.gaussian_blur(
            image, kernel_size=3, sigma=(0.1, 0.8)
        )

    return image, mask


def paired_val_transform(image, mask):
    image = TF.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.BILINEAR,
    )

    mask = TF.resize(
        mask,
        [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.NEAREST,
    )

    return image, mask


class MarsDataset(Dataset):

    def __init__(self, pairs, train=True):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        if self.train:
            image, mask = paired_train_transform(
                image, mask
            )
        else:
            image, mask = paired_val_transform(
                image, mask
            )

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            IMAGENET_MEAN,
            IMAGENET_STD
        )

        mask = torch.from_numpy(
            np.asarray(
                mask,
                dtype=np.int64
            )
        )

        return image, mask, str(image_path)


train_ds = MarsDataset(
    train_pairs,
    train=True
)

val_ds = MarsDataset(
    val_pairs,
    train=False
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=shuffle,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)


# ============================================================
# 10. Y-Net building blocks
# ============================================================
class ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        p=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                k,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AGCB(nn.Module):
    """
    Adaptive Gated Convolutional Block.

    The block creates a learned gate over convolutional features
    and uses a residual connection to preserve useful information.
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.out_act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)

        y = self.conv2(y)
        y = self.bn2(y)

        gate = self.gate(y)

        y = y * gate

        return self.out_act(
            x + y
        )


class CSAIB(nn.Module):
    """
    Channel-wise and Spatial Attention Integration Block.
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=False,
            ),
        )

        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(
            x, 1
        )

        mx = F.adaptive_max_pool2d(
            x, 1
        )

        channel_attn = torch.sigmoid(
            self.mlp(avg) +
            self.mlp(mx)
        )

        x = x * channel_attn

        avg_sp = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_sp, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        spatial_input = torch.cat(
            [avg_sp, max_sp],
            dim=1,
        )

        spatial_attn = torch.sigmoid(
            self.spatial(
                spatial_input
            )
        )

        return x * spatial_attn


class FARB(nn.Module):
    """
    Fusion Attention Refinement Block.

    It fuses decoder and skip features, then refines the fused
    representation using CSAIB and a residual path.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
    ):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.attn = CSAIB(
            out_ch
        )

        self.residual = (
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False,
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)

        y = self.fuse(x)
        y = self.attn(y)

        return F.relu(
            y + identity,
            inplace=True,
        )


# ============================================================
# 11. EfficientNet encoder
# ============================================================
class EfficientNetEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        # If pretrained weights cannot be downloaded, fall back to
        # randomly initialized EfficientNet instead of crashing.
        try:
            backbone = efficientnet_b0(weights=weights)
        except Exception as exc:
            print("WARNING: Could not load pretrained EfficientNet-B0 weights.")
            print("Reason:", exc)
            print("Falling back to randomly initialized EfficientNet-B0.")
            backbone = efficientnet_b0(weights=None)

        self.features = backbone.features

    def forward(self, x):
        skips = []

        for i, layer in enumerate(
            self.features
        ):
            x = layer(x)

            # For 512x512:
            # index 0 -> 256x256, 32 ch
            # index 2 -> 128x128, 24 ch
            # index 3 -> 64x64, 40 ch
            # index 4 -> 32x32, 80 ch
            # final -> 16x16, 1280 ch
            if i in (0, 2, 3, 4):
                skips.append(x)

        return skips, x


# ============================================================
# 12. Decoder
# ============================================================
class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
    ):
        super().__init__()

        self.farb = FARB(
            in_ch + skip_ch,
            out_ch,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.farb(x)


class YNet(nn.Module):

    def __init__(
        self,
        num_classes=12,
        pretrained=True,
    ):
        super().__init__()

        self.encoder = EfficientNetEncoder(
            pretrained=pretrained
        )

        # Encoder feature enhancement.
        self.agcb0 = AGCB(32)
        self.csai0 = CSAIB(32)

        self.agcb1 = AGCB(24)
        self.csai1 = CSAIB(24)

        self.agcb2 = AGCB(40)
        self.csai2 = CSAIB(40)

        self.agcb3 = AGCB(80)
        self.csai3 = CSAIB(80)

        # Bottleneck.
        self.bottleneck_reduce = ConvBNReLU(
            1280,
            256,
            k=1,
            p=0,
        )

        self.bottleneck_agcb = AGCB(
            256
        )

        self.bottleneck_csai = CSAIB(
            256
        )

        self.bottleneck_farb = FARB(
            256,
            256,
        )

        # Decoder.
        self.dec1 = DecoderBlock(
            256, 80, 192
        )

        self.dec2 = DecoderBlock(
            192, 40, 128
        )

        self.dec3 = DecoderBlock(
            128, 24, 64
        )

        self.dec4 = DecoderBlock(
            64, 32, 32
        )

        self.dec5 = nn.Sequential(
            ConvBNReLU(32, 32),
            ConvBNReLU(32, 16),
        )

        # Y-Net branch 1: terrain segmentation.
        self.terrain_head = nn.Conv2d(
            16,
            num_classes,
            1,
        )

        # Y-Net branch 2: boundary refinement.
        self.boundary_head = nn.Conv2d(
            16,
            1,
            1,
        )

    def forward(self, x):
        skips, bottleneck = self.encoder(x)

        s0 = self.csai0(
            self.agcb0(skips[0])
        )

        s1 = self.csai1(
            self.agcb1(skips[1])
        )

        s2 = self.csai2(
            self.agcb2(skips[2])
        )

        s3 = self.csai3(
            self.agcb3(skips[3])
        )

        x = self.bottleneck_reduce(
            bottleneck
        )

        x = self.bottleneck_agcb(x)
        x = self.bottleneck_csai(x)
        x = self.bottleneck_farb(x)

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        x = self.dec4(x, s0)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.dec5(x)

        terrain_logits = self.terrain_head(x)
        boundary_logits = self.boundary_head(x)

        return terrain_logits, boundary_logits


model = YNet(
    num_classes=NUM_CLASSES,
    pretrained=USE_PRETRAINED,
).to(DEVICE)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nY-Net parameters:",
    f"{num_params / 1e6:.2f} M"
)


# ============================================================
# 13. Boundary target
# ============================================================
def make_boundary_target(mask):
    """
    Boundary = pixel whose label differs from at least one
    4-connected neighbor.
    """

    m = mask.unsqueeze(1)

    left = F.pad(
        m[:, :, :, :-1],
        (1, 0, 0, 0),
        value=255,
    )

    right = F.pad(
        m[:, :, :, 1:],
        (0, 1, 0, 0),
        value=255,
    )

    up = F.pad(
        m[:, :, :-1, :],
        (0, 0, 1, 0),
        value=255,
    )

    down = F.pad(
        m[:, :, 1:, :],
        (0, 0, 0, 1),
        value=255,
    )

    boundary = (
        (m != left) |
        (m != right) |
        (m != up) |
        (m != down)
    ).float()

    boundary[:, :, 0, :] = 0
    boundary[:, :, -1, :] = 0
    boundary[:, :, :, 0] = 0
    boundary[:, :, :, -1] = 0

    return boundary


# ============================================================
# 14. Losses
# ============================================================
class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights, smooth=1.0):
        super().__init__()
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        one_hot = F.one_hot(
            target.clamp(0, NUM_CLASSES - 1),
            NUM_CLASSES
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        weights = self.class_weights / self.class_weights.mean()

        return 1.0 - (dice * weights).sum() / weights.sum()


class FocalLoss(nn.Module):
    """Multiclass focal loss that emphasizes hard pixels."""
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "weight",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        focal = -(1.0 - pt).pow(self.gamma) * logpt
        alpha = self.weight[target]
        return (alpha * focal).mean()


class TverskyLoss(nn.Module):
    """Multiclass Tversky loss for overlap and class-imbalance control."""
    def __init__(self, class_weights, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(
            target.clamp(0, NUM_CLASSES - 1),
            NUM_CLASSES
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dims)
        fp = (probs * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - probs) * one_hot).sum(dims)

        score = (
            tp + self.smooth
        ) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        weights = self.class_weights / self.class_weights.mean()
        return 1.0 - (score * weights).sum() / weights.sum()


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        p = probs.reshape(-1)
        t = target.float().reshape(-1)

        intersection = (p * t).sum()

        return 1.0 - (
            2.0 * intersection + self.smooth
        ) / (
            p.sum() + t.sum() + self.smooth
        )


class YNetLoss(nn.Module):
    """Stable overlap-focused segmentation loss."""
    def __init__(self, class_weights):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights, dtype=torch.float32
            ),
            label_smoothing=0.02,
        )

        self.dice = WeightedDiceLoss(class_weights)
        self.focal = FocalLoss(class_weights, gamma=2.0)
        self.tversky = TverskyLoss(
            class_weights,
            alpha=0.30,
            beta=0.70,
        )
        self.boundary_bce = nn.BCEWithLogitsLoss()
        self.boundary_dice = BinaryDiceLoss()

    def forward(
        self,
        terrain_logits,
        boundary_logits,
        target,
    ):
        boundary_target = make_boundary_target(target)

        ce_loss = self.ce(terrain_logits, target)
        dice_loss = self.dice(terrain_logits, target)
        focal_loss = self.focal(terrain_logits, target)
        tversky_loss = self.tversky(terrain_logits, target)

        bce = self.boundary_bce(
            boundary_logits, boundary_target
        )
        bdice = self.boundary_dice(
            boundary_logits, boundary_target
        )
        boundary_loss = 0.5 * bce + 0.5 * bdice

        total = (
            0.70 * ce_loss +
            1.00 * dice_loss +
            0.60 * focal_loss +
            0.80 * tversky_loss +
            0.20 * boundary_loss
        )

        return total


criterion = YNetLoss(class_weights).to(DEVICE)


# ============================================================
# 15. Metrics
# ============================================================
@torch.no_grad()
def update_confusion_matrix(
    confmat,
    pred,
    target,
    num_classes,
):
    pred = (
        pred.reshape(-1)
        .cpu()
        .numpy()
    )

    target = (
        target.reshape(-1)
        .cpu()
        .numpy()
    )

    valid = (
        (target >= 0) &
        (target < num_classes)
    )

    target = target[valid]
    pred = pred[valid]

    encoded = (
        target * num_classes +
        pred
    )

    binc = np.bincount(
        encoded,
        minlength=num_classes ** 2,
    )

    confmat += binc.reshape(
        num_classes,
        num_classes,
    )


def metrics_from_confmat(confmat):
    tp = np.diag(
        confmat
    ).astype(np.float64)

    fp = (
        confmat.sum(axis=0) -
        tp
    )

    fn = (
        confmat.sum(axis=1) -
        tp
    )

    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    denom_prec = tp + fp
    denom_rec = tp + fn

    iou = np.divide(
        tp,
        denom_iou,
        out=np.zeros_like(tp),
        where=denom_iou > 0,
    )

    dice = np.divide(
        2 * tp,
        denom_dice,
        out=np.zeros_like(tp),
        where=denom_dice > 0,
    )

    precision = np.divide(
        tp,
        denom_prec,
        out=np.zeros_like(tp),
        where=denom_prec > 0,
    )

    recall = np.divide(
        tp,
        denom_rec,
        out=np.zeros_like(tp),
        where=denom_rec > 0,
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    pixel_acc = (
        tp.sum() /
        max(confmat.sum(), 1)
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pixel_acc": pixel_acc,
        "miou_all": iou.mean(),
        "mdice_all": dice.mean(),
        "miou_fg": iou[1:].mean(),
        "mdice_fg": dice[1:].mean(),
        "rare_miou": (
            float(np.mean(iou[rare_classes]))
            if rare_classes
            else 0.0
        ),
    }


# ============================================================
# 16. Optimizer / scheduler / AMP
# ============================================================
# Differential learning rates: keep pretrained EfficientNet features
# stable while allowing the new segmentation decoder to learn faster.
encoder_params = list(model.encoder.parameters())
encoder_ids = {id(p) for p in encoder_params}
decoder_params = [
    p for p in model.parameters()
    if id(p) not in encoder_ids
]

optimizer = torch.optim.AdamW(
    [
        {"params": encoder_params, "lr": ENCODER_LR},
        {"params": decoder_params, "lr": LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(AMP and DEVICE.type == "cuda"),
)


# ============================================================
# 17. Training
# ============================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    model.train()

    running_loss = 0.0
    n = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks, _) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(AMP and DEVICE.type == "cuda"),
        ):
            terrain_logits, boundary_logits = model(images)
            loss = criterion(
                terrain_logits,
                boundary_logits,
                masks,
            )
            loss_for_backward = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss_for_backward).backward()

        if (
            (step + 1) % GRAD_ACCUM_STEPS == 0
            or (step + 1) == len(loader)
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(n, 1)


# ============================================================
# 18. Validation
# ============================================================
@torch.no_grad()
def predict_with_tta(model, images):
    """Average logits from original + H/V flips + 180 rotation."""
    variants = [
        (images, None),
        (torch.flip(images, dims=[3]), "h"),
        (torch.flip(images, dims=[2]), "v"),
        (torch.rot90(images, 2, dims=[2, 3]), "r"),
    ]

    logits_sum = None

    for aug, kind in variants:
        terrain_logits, _ = model(aug)

        if kind == "h":
            terrain_logits = torch.flip(
                terrain_logits, dims=[3]
            )
        elif kind == "v":
            terrain_logits = torch.flip(
                terrain_logits, dims=[2]
            )
        elif kind == "r":
            terrain_logits = torch.rot90(
                terrain_logits, 2, dims=[2, 3]
            )

        logits_sum = (
            terrain_logits
            if logits_sum is None
            else logits_sum + terrain_logits
        )

    return logits_sum / len(variants)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    use_tta=False,
):
    model.eval()

    running_loss = 0.0
    n = 0

    confmat = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    for images, masks, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # Loss is evaluated on the normal forward pass.
        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(AMP and DEVICE.type == "cuda"),
        ):
            terrain_logits, boundary_logits = model(images)

            loss = criterion(
                terrain_logits,
                boundary_logits,
                masks,
            )

            if use_tta:
                terrain_logits = predict_with_tta(
                    model, images
                )

        pred = torch.argmax(
            terrain_logits, dim=1
        )

        update_confusion_matrix(
            confmat,
            pred,
            masks,
            NUM_CLASSES,
        )

        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs

    metrics = metrics_from_confmat(confmat)
    metrics["loss"] = running_loss / max(n, 1)
    metrics["confmat"] = confmat

    return metrics


# ============================================================
# 19. Pre-training smoke test
# ============================================================
# Run one real batch before starting the long training loop.
# This catches shape/device/loss errors immediately.
print("\n================ PRE-TRAINING CHECK ================")
try:
    test_images, test_masks, _ = next(iter(train_loader))
    test_images = test_images.to(DEVICE)
    test_masks = test_masks.to(DEVICE)
    model.train()
    with torch.amp.autocast(
        device_type="cuda",
        enabled=(AMP and DEVICE.type == "cuda"),
    ):
        test_terrain, test_boundary = model(test_images)
        test_loss = criterion(test_terrain, test_boundary, test_masks)
    if not torch.isfinite(test_loss):
        raise RuntimeError(f"Pre-training loss is not finite: {test_loss.item()}")
    if test_terrain.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Model returned {test_terrain.shape[1]} classes; expected {NUM_CLASSES}."
        )
    print("Smoke test PASSED")
    print("Input shape    :", tuple(test_images.shape))
    print("Output shape   :", tuple(test_terrain.shape))
    print("Initial loss   :", f"{test_loss.item():.6f}")
except Exception as exc:
    raise RuntimeError(
        "Pre-training check failed. Training was NOT started. "
        f"Original error: {exc}"
    ) from exc


# ============================================================
# 20. Main training loop
# ============================================================
best_score = -float("inf")
bad_epochs = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "miou": [],
    "dice": [],
    "rare_miou": [],
}

print(
    "\n================ START TRAINING ================"
)

for epoch in range(
    1,
    EPOCHS + 1,
):
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    val_metrics = validate(
        model,
        val_loader,
        criterion,
        use_tta=False,
    )

    scheduler.step()

    miou = val_metrics["miou_all"]
    miou_fg = val_metrics["miou_fg"]
    mdice = val_metrics["mdice_fg"]
    rare_miou = val_metrics["rare_miou"]

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_metrics["loss"]
    )

    history["miou"].append(
        miou
    )

    history["dice"].append(
        mdice
    )

    history["rare_miou"].append(
        rare_miou
    )

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_metrics['loss']:.4f} | "
        f"mIoU-ALL {miou:.4f} | "
        f"mIoU-FG {miou_fg:.4f} | "
        f"Dice-FG {mdice:.4f} | "
        f"Rare-mIoU {rare_miou:.4f} | "
        f"LR {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Give rare-class performance equal importance to overall FG mIoU.
    score = (
        0.70 * float(miou) +
        0.20 * float(miou_fg) +
        0.10 * float(rare_miou)
    )

    if not np.isfinite(score):
        score = -1e9

    if score > best_score:
        best_score = score
        bad_epochs = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "class_weights": class_weights,
                "palette_rgb": PALETTE_RGB,
                "class_names": CLASS_NAMES,
                "history": history,
            },
            CHECKPOINT_PATH,
        )

        print(
            "  ✓ Saved best checkpoint:",
            CHECKPOINT_PATH,
        )

    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print(
            "Early stopping."
        )
        break


# ============================================================
# 20. Final evaluation
# ============================================================
if not os.path.exists(CHECKPOINT_PATH):
    raise RuntimeError(
        "No checkpoint was created. Training did not produce a valid model."
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

final_metrics = validate(
    model,
    val_loader,
    criterion,
    use_tta=USE_TTA,
)

print(
    "\n================ FINAL RESULTS ================"
)
print("Final evaluation uses TTA:", USE_TTA)

print(
    f"Pixel Accuracy : "
    f"{final_metrics['pixel_acc']:.4f}"
)

print(
    f"mIoU (all)     : "
    f"{final_metrics['miou_all']:.4f}"
)

print(
    f"mIoU (FG)      : "
    f"{final_metrics['miou_fg']:.4f}"
)

print(
    f"Mean Dice (FG) : "
    f"{final_metrics['mdice_fg']:.4f}"
)

print(
    f"Rare-class mIoU: "
    f"{final_metrics['rare_miou']:.4f}"
)

print(
    "\nPer-class metrics:"
)

for c in range(NUM_CLASSES):
    print(
        f"{c:2d} {CLASS_NAMES[c]:10s} | "
        f"IoU={final_metrics['iou'][c]:.4f} | "
        f"Dice={final_metrics['dice'][c]:.4f} | "
        f"Prec={final_metrics['precision'][c]:.4f} | "
        f"Rec={final_metrics['recall'][c]:.4f} | "
        f"F1={final_metrics['f1'][c]:.4f}"
    )

print(
    "\nConfusion matrix:"
)

print(
    final_metrics["confmat"]
)


# ============================================================
# 21. Save prediction visualizations
# ============================================================
def denormalize(x):
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=x.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        device=x.device,
    ).view(1, 3, 1, 1)

    return (
        x * std + mean
    ).clamp(0, 1)


def labels_to_rgb(labels):
    pal = torch.tensor(
        PALETTE_RGB,
        dtype=torch.uint8,
        device=labels.device,
    )

    rgb = pal[
        labels.long()
    ]

    return (
        rgb.permute(0, 3, 1, 2)
        .float() / 255.0
    )


@torch.no_grad()
def save_predictions(
    model,
    loader,
    max_batches=5,
):
    model.eval()

    saved = 0

    for b, (
        images,
        masks,
        paths,
    ) in enumerate(loader):

        if b >= max_batches:
            break

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        terrain_logits, boundary_logits = (
            model(images)
        )

        preds = terrain_logits.argmax(
            dim=1
        )

        boundary = torch.sigmoid(
            boundary_logits
        )

        original = denormalize(
            images
        )

        gt_rgb = labels_to_rgb(
            masks
        )

        pred_rgb = labels_to_rgb(
            preds
        )

        boundary_rgb = boundary.repeat(
            1, 3, 1, 1
        )

        for i in range(
            images.size(0)
        ):
            base = Path(
                paths[i]
            ).stem

            save_image(
                original[i],
                os.path.join(
                    PRED_DIR,
                    f"{base}_image.png",
                ),
            )

            save_image(
                gt_rgb[i],
                os.path.join(
                    PRED_DIR,
                    f"{base}_gt.png",
                ),
            )

            save_image(
                pred_rgb[i],
                os.path.join(
                    PRED_DIR,
                    f"{base}_pred.png",
                ),
            )

            save_image(
                boundary_rgb[i],
                os.path.join(
                    PRED_DIR,
                    f"{base}_boundary.png",
                ),
            )

            saved += 1

    print(
        f"Saved {saved} prediction sets to "
        f"{PRED_DIR}"
    )


save_predictions(
    model,
    val_loader,
    max_batches=10,
)


# ============================================================
# 22. Optional model summary
# ============================================================
try:
    from torchinfo import summary

    summary(
        model,
        input_size=(
            1,
            3,
            IMG_SIZE,
            IMG_SIZE,
        ),
        depth=4,
    )

except Exception as exc:
    print("\nTorchinfo summary skipped (optional):", exc)

print("\nDone.")
print("NOTE: 0.90 mIoU cannot be guaranteed by code alone; it depends on dataset quality, label consistency, class separability, and the validation split.")
print(
    "Best checkpoint:",
    CHECKPOINT_PATH
)
print(
    "Predictions:",
    PRED_DIR
)
# ============================================================
# ADD-ON ONLY
# EXTENDED METRICS + STANDARD DEVIATION + BENCHMARKING
# ============================================================
#
# IMPORTANT:
# This block DOES NOT modify your existing Y-Net model,
# training, validation, loss, or final_metrics.
#
# Your ORIGINAL means remain the official means:
#     final_metrics["miou_all"]
#     final_metrics["miou_fg"]
#     final_metrics["mdice_fg"]
#     final_metrics["pixel_acc"]
#     final_metrics["rare_miou"]
#
# This block ONLY adds:
#     SD for metrics
#     Sensitivity
#     Specificity
#     PQ proxy
#     FLOPs
#     Inference time
#     FPS
#     CPU usage
#     RAM usage
#     30 raw/GT/predicted/overlay images
#
# ============================================================
# FLOPs package
try:
    from thop import profile
except ImportError:
    profile = None
    print("\nTHOP not installed.")
    print("Install with:")
    print("pip install thop psutil")


# ============================================================
# SETTINGS
# ============================================================

EXTENDED_RESULTS_DIR = os.path.join(
    PRED_DIR,
    "extended_results"
)

os.makedirs(
    EXTENDED_RESULTS_DIR,
    exist_ok=True
)

NUM_VISUAL_IMAGES = 30

# Number of repeated forward passes for
# inference-time/FPS/CPU measurements.
BENCHMARK_RUNS = 100

WARMUP_RUNS = 20


# ============================================================
# MODEL
# ============================================================

model.eval()

print()
print("=" * 80)
print("EXTENDED EVALUATION — ORIGINAL RESULTS PRESERVED")
print("=" * 80)


# ============================================================
# STORAGE
# ============================================================

per_image_miou = []
per_image_miou_fg = []

per_image_dice = []
per_image_dice_fg = []

per_image_precision = []
per_image_recall = []
per_image_f1 = []

per_image_sensitivity = []
per_image_specificity = []

per_image_pq = []


# ============================================================
# COMPUTATIONAL METRICS STORAGE
# ============================================================

inference_times_ms = []
fps_values = []
cpu_values = []
memory_values_mb = []


# ============================================================
# PROCESS
# ============================================================

process = psutil.Process(
    os.getpid()
)


# ============================================================
# PER-IMAGE METRIC FUNCTION
# ============================================================

def extended_image_metrics(
    prediction,
    target,
    num_classes
):

    prediction = np.asarray(
        prediction,
        dtype=np.int64
    )

    target = np.asarray(
        target,
        dtype=np.int64
    )

    valid = (
        (target >= 0)
        &
        (target < num_classes)
    )

    prediction = prediction[valid]
    target = target[valid]

    if target.size == 0:

        return {
            "miou_all": 0.0,
            "miou_fg": 0.0,
            "dice_all": 0.0,
            "dice_fg": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "pq": 0.0,
        }


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    encoded = (
        target * num_classes
        + prediction
    )

    cm = np.bincount(
        encoded,
        minlength=num_classes ** 2
    ).reshape(
        num_classes,
        num_classes
    ).astype(
        np.float64
    )


    tp = np.diag(cm)

    fp = (
        cm.sum(axis=0)
        - tp
    )

    fn = (
        cm.sum(axis=1)
        - tp
    )

    tn = (
        cm.sum()
        - tp
        - fp
        - fn
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_den = (
        tp
        + fp
        + fn
    )

    iou = np.divide(
        tp,
        iou_den,
        out=np.zeros_like(tp),
        where=iou_den > 0
    )


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_den = (
        2 * tp
        + fp
        + fn
    )

    dice = np.divide(
        2 * tp,
        dice_den,
        out=np.zeros_like(tp),
        where=dice_den > 0
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_den = (
        tp
        + fp
    )

    precision = np.divide(
        tp,
        precision_den,
        out=np.zeros_like(tp),
        where=precision_den > 0
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_den = (
        tp
        + fn
    )

    recall = np.divide(
        tp,
        recall_den,
        out=np.zeros_like(tp),
        where=recall_den > 0
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1_den = (
        precision
        + recall
    )

    f1 = np.divide(
        2 * precision * recall,
        f1_den,
        out=np.zeros_like(tp),
        where=f1_den > 0
    )


    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity_den = (
        tn
        + fp
    )

    specificity = np.divide(
        tn,
        specificity_den,
        out=np.zeros_like(tp),
        where=specificity_den > 0
    )


    # --------------------------------------------------------
    # Valid classes
    # --------------------------------------------------------

    valid_classes = (
        iou_den > 0
    )


    if np.any(valid_classes):

        miou_all = float(
            np.mean(
                iou[valid_classes]
            )
        )

        dice_all = float(
            np.mean(
                dice[valid_classes]
            )
        )

        precision_mean = float(
            np.mean(
                precision[valid_classes]
            )
        )

        recall_mean = float(
            np.mean(
                recall[valid_classes]
            )
        )

        f1_mean = float(
            np.mean(
                f1[valid_classes]
            )
        )

        sensitivity_mean = float(
            np.mean(
                recall[valid_classes]
            )
        )

        specificity_mean = float(
            np.mean(
                specificity[valid_classes]
            )
        )

    else:

        miou_all = 0.0
        dice_all = 0.0
        precision_mean = 0.0
        recall_mean = 0.0
        f1_mean = 0.0
        sensitivity_mean = 0.0
        specificity_mean = 0.0


    # --------------------------------------------------------
    # Foreground metrics
    # Same class convention as your original code:
    # class 0 = background
    # --------------------------------------------------------

    if num_classes > 1:

        foreground_valid = (
            valid_classes[1:]
        )

        if np.any(foreground_valid):

            miou_fg = float(
                np.mean(
                    iou[1:][foreground_valid]
                )
            )

            dice_fg = float(
                np.mean(
                    dice[1:][foreground_valid]
                )
            )

        else:

            miou_fg = 0.0
            dice_fg = 0.0

    else:

        miou_fg = miou_all
        dice_fg = dice_all


    # ========================================================
    # SEMANTIC PQ PROXY
    #
    # This is NOT conventional instance-level PQ because
    # your dataset contains semantic masks, not instance IDs.
    #
    # PQ proxy = SQ × RQ
    #
    # SQ = mean IoU
    # RQ = mean F1
    # ========================================================

    sq = miou_all
    rq = f1_mean

    pq = (
        sq * rq
    )


    return {
        "miou_all": miou_all,
        "miou_fg": miou_fg,
        "dice_all": dice_all,
        "dice_fg": dice_fg,
        "precision": precision_mean,
        "recall": recall_mean,
        "f1": f1_mean,
        "sensitivity": sensitivity_mean,
        "specificity": specificity_mean,
        "pq": pq,
    }


# ============================================================
# IMAGE VISUALIZATION HELPERS
# ============================================================

def extended_denormalize(x):

    mean = torch.tensor(
        IMAGENET_MEAN,
        device=x.device
    ).view(
        1, 3, 1, 1
    )

    std = torch.tensor(
        IMAGENET_STD,
        device=x.device
    ).view(
        1, 3, 1, 1
    )

    return (
        x * std + mean
    ).clamp(
        0,
        1
    )


def extended_labels_to_rgb(
    labels
):

    pal = torch.tensor(
        PALETTE_RGB,
        dtype=torch.uint8,
        device=labels.device
    )

    rgb = pal[
        labels.long()
    ]

    return (
        rgb.permute(
            0,
            3,
            1,
            2
        ).float()
        / 255.0
    )


# ============================================================
# SAVE 30 RAW / GT / PRED / OVERLAY
# ============================================================

visual_saved = 0


print()
print(
    "Generating 30 qualitative results..."
)


with torch.no_grad():

    for images, masks, paths in val_loader:

        if visual_saved >= NUM_VISUAL_IMAGES:
            break


        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )


        terrain_logits, boundary_logits = (
            model(images)
        )


        predictions = torch.argmax(
            terrain_logits,
            dim=1
        )


        original = extended_denormalize(
            images
        )

        gt_rgb = extended_labels_to_rgb(
            masks
        )

        pred_rgb = extended_labels_to_rgb(
            predictions
        )


        for i in range(
            images.size(0)
        ):

            if visual_saved >= NUM_VISUAL_IMAGES:
                break


            # ------------------------------------------------
            # RGB arrays
            # ------------------------------------------------

            raw = (
                original[i]
                .detach()
                .cpu()
                .permute(
                    1,
                    2,
                    0
                )
                .numpy()
            )


            gt = (
                masks[i]
                .detach()
                .cpu()
                .numpy()
            )


            pred = (
                predictions[i]
                .detach()
                .cpu()
                .numpy()
            )


            pred_color = (
                pred_rgb[i]
                .detach()
                .cpu()
                .permute(
                    1,
                    2,
                    0
                )
                .numpy()
            )


            raw = np.clip(
                raw,
                0,
                1
            )


            # ------------------------------------------------
            # Overlay prediction on raw image
            # ------------------------------------------------

            overlay = (
                0.55 * raw
                +
                0.45 * pred_color
            )

            overlay = np.clip(
                overlay,
                0,
                1
            )


            # ------------------------------------------------
            # Save raw image
            # ------------------------------------------------

            plt.imsave(
                os.path.join(
                    EXTENDED_RESULTS_DIR,
                    f"{visual_saved + 1:02d}_raw.png"
                ),
                raw
            )


            # ------------------------------------------------
            # Save ground truth
            # ------------------------------------------------

            plt.imsave(
                os.path.join(
                    EXTENDED_RESULTS_DIR,
                    f"{visual_saved + 1:02d}_ground_truth.png"
                ),
                gt,
                cmap="tab20"
            )


            # ------------------------------------------------
            # Save predicted image
            # ------------------------------------------------

            plt.imsave(
                os.path.join(
                    EXTENDED_RESULTS_DIR,
                    f"{visual_saved + 1:02d}_predicted.png"
                ),
                pred,
                cmap="tab20"
            )


            # ------------------------------------------------
            # Save overlay
            # ------------------------------------------------

            plt.imsave(
                os.path.join(
                    EXTENDED_RESULTS_DIR,
                    f"{visual_saved + 1:02d}_overlay.png"
                ),
                overlay
            )


            visual_saved += 1


print(
    f"Saved {visual_saved} complete "
    "visualization sets."
)

print(
    "Location:",
    os.path.abspath(
        EXTENDED_RESULTS_DIR
    )
)


# ============================================================
# PER-IMAGE METRICS
# ============================================================

print()
print(
    "Calculating per-image statistics..."
)


with torch.no_grad():

    for images, masks, paths in val_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )


        terrain_logits, boundary_logits = (
            model(images)
        )


        predictions = torch.argmax(
            terrain_logits,
            dim=1
        )


        for i in range(
            images.size(0)
        ):

            result = extended_image_metrics(
                predictions[i]
                .detach()
                .cpu()
                .numpy(),

                masks[i]
                .detach()
                .cpu()
                .numpy(),

                NUM_CLASSES
            )


            per_image_miou.append(
                result["miou_all"]
            )

            per_image_miou_fg.append(
                result["miou_fg"]
            )

            per_image_dice.append(
                result["dice_all"]
            )

            per_image_dice_fg.append(
                result["dice_fg"]
            )

            per_image_precision.append(
                result["precision"]
            )

            per_image_recall.append(
                result["recall"]
            )

            per_image_f1.append(
                result["f1"]
            )

            per_image_sensitivity.append(
                result["sensitivity"]
            )

            per_image_specificity.append(
                result["specificity"]
            )

            per_image_pq.append(
                result["pq"]
            )


# ============================================================
# STANDARD DEVIATION FUNCTION
# ============================================================

def get_sd(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1
        )
    )


# ============================================================
# SD VALUES
# ============================================================

miou_all_sd = get_sd(
    per_image_miou
)

miou_fg_sd = get_sd(
    per_image_miou_fg
)

dice_all_sd = get_sd(
    per_image_dice
)

dice_fg_sd = get_sd(
    per_image_dice_fg
)

precision_sd = get_sd(
    per_image_precision
)

recall_sd = get_sd(
    per_image_recall
)

f1_sd = get_sd(
    per_image_f1
)

sensitivity_sd = get_sd(
    per_image_sensitivity
)

specificity_sd = get_sd(
    per_image_specificity
)

pq_sd = get_sd(
    per_image_pq
)


# ============================================================
# FLOPs
# ============================================================

gflops = None


if profile is not None:

    try:

        dummy_input = torch.randn(
            1,
            3,
            IMG_SIZE,
            IMG_SIZE,
            device=DEVICE
        )


        macs, _ = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )


        # THOP reports MACs.
        # FLOPs ≈ 2 × MACs.

        gflops = (
            2.0
            * macs
            / 1e9
        )

    except Exception as e:

        print(
            "\nFLOPs calculation failed:"
        )

        print(e)


# ============================================================
# INFERENCE BENCHMARK
# ============================================================

print()
print(
    "Benchmarking inference..."
)


benchmark_input = torch.randn(
    1,
    3,
    IMG_SIZE,
    IMG_SIZE,
    device=DEVICE
)


# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

with torch.no_grad():

    for _ in range(
        WARMUP_RUNS
    ):

        _ = model(
            benchmark_input
        )


if torch.cuda.is_available():

    torch.cuda.synchronize()


# ------------------------------------------------------------
# Repeated inference
# ------------------------------------------------------------

for _ in range(
    BENCHMARK_RUNS
):

    # CPU measurement before inference
    process.cpu_percent(
        interval=None
    )


    if torch.cuda.is_available():

        torch.cuda.synchronize()


    start = time.perf_counter()


    with torch.no_grad():

        _ = model(
            benchmark_input
        )


    if torch.cuda.is_available():

        torch.cuda.synchronize()


    end = time.perf_counter()


    elapsed_ms = (
        end - start
    ) * 1000.0


    inference_times_ms.append(
        elapsed_ms
    )


    # --------------------------------------------------------
    # FPS for this run
    # --------------------------------------------------------

    if elapsed_ms > 0:

        fps_values.append(
            1000.0
            / elapsed_ms
        )


    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_values.append(
        process.cpu_percent(
            interval=None
        )
    )


    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    memory_values_mb.append(
        process.memory_info().rss
        / (1024 ** 2)
    )


# ============================================================
# COMPUTATIONAL MEANS + SD
# ============================================================

inference_mean = float(
    np.mean(
        inference_times_ms
    )
)

inference_sd = get_sd(
    inference_times_ms
)


fps_mean = float(
    np.mean(
        fps_values
    )
)

fps_sd = get_sd(
    fps_values
)


cpu_mean = float(
    np.mean(
        cpu_values
    )
)

cpu_sd = get_sd(
    cpu_values
)


memory_mean = float(
    np.mean(
        memory_values_mb
    )
)

memory_sd = get_sd(
    memory_values_mb
)


# ============================================================
# GPU MEMORY
# ============================================================

gpu_memory_mb = None


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        DEVICE
    )


    with torch.no_grad():

        _ = model(
            benchmark_input
        )


    torch.cuda.synchronize()


    gpu_memory_mb = (
        torch.cuda.max_memory_allocated(
            DEVICE
        )
        / (1024 ** 2)
    )


# ============================================================
# FINAL REPORT
# ============================================================
#
# CRITICAL:
# The following means for the original metrics are taken
# DIRECTLY from final_metrics.
#
# Therefore your existing mIoU result is NOT replaced.
#
# ============================================================

print()
print("=" * 80)
print("FINAL EXTENDED RESULTS")
print("=" * 80)

print()

# ------------------------------------------------------------
# ORIGINAL METRICS — MEANS PRESERVED
# ------------------------------------------------------------

print(
    "ORIGINAL METRICS + SD"
)

print("-" * 80)


print(
    f"Pixel Accuracy : "
    f"{final_metrics['pixel_acc']:.4f}"
)


print(
    f"mIoU (all)     : "
    f"{final_metrics['miou_all']:.4f}"
    f" ± {miou_all_sd:.4f}"
)


print(
    f"mIoU (FG)      : "
    f"{final_metrics['miou_fg']:.4f}"
    f" ± {miou_fg_sd:.4f}"
)


print(
    f"Mean Dice (FG) : "
    f"{final_metrics['mdice_fg']:.4f}"
    f" ± {dice_fg_sd:.4f}"
)


print(
    f"Rare-class mIoU: "
    f"{final_metrics['rare_miou']:.4f}"
)


# ------------------------------------------------------------
# ADDITIONAL METRICS
# ------------------------------------------------------------

print()
print(
    "ADDITIONAL SEGMENTATION METRICS"
)

print("-" * 80)


# These are macro means from the same per-class arrays
# already produced by your original metrics_from_confmat().

existing_precision_mean = float(
    np.mean(
        final_metrics["precision"]
    )
)

existing_recall_mean = float(
    np.mean(
        final_metrics["recall"]
    )
)

existing_f1_mean = float(
    np.mean(
        final_metrics["f1"]
    )
)

existing_dice_all_mean = float(
    final_metrics["mdice_all"]
)


print(
    f"Precision      : "
    f"{existing_precision_mean:.4f}"
    f" ± {precision_sd:.4f}"
)


print(
    f"Recall         : "
    f"{existing_recall_mean:.4f}"
    f" ± {recall_sd:.4f}"
)


print(
    f"F1 Score       : "
    f"{existing_f1_mean:.4f}"
    f" ± {f1_sd:.4f}"
)


print(
    f"Dice (all)     : "
    f"{existing_dice_all_mean:.4f}"
    f" ± {dice_all_sd:.4f}"
)


print(
    f"Sensitivity    : "
    f"{existing_recall_mean:.4f}"
    f" ± {sensitivity_sd:.4f}"
)


# ------------------------------------------------------------
# SPECIFICITY
# ------------------------------------------------------------

# Dataset-level specificity from the final confusion matrix.

cm = final_metrics[
    "confmat"
].astype(
    np.float64
)

tp = np.diag(cm)

fp = (
    cm.sum(axis=0)
    - tp
)

fn = (
    cm.sum(axis=1)
    - tp
)

tn = (
    cm.sum()
    - tp
    - fp
    - fn
)

specificity_den = (
    tn + fp
)

specificity_per_class = np.divide(
    tn,
    specificity_den,
    out=np.zeros_like(tn),
    where=specificity_den > 0
)

valid_specificity_classes = (
    specificity_den > 0
)

if np.any(
    valid_specificity_classes
):

    existing_specificity_mean = float(
        np.mean(
            specificity_per_class[
                valid_specificity_classes
            ]
        )
    )

else:

    existing_specificity_mean = 0.0


print(
    f"Specificity    : "
    f"{existing_specificity_mean:.4f}"
    f" ± {specificity_sd:.4f}"
)


# ------------------------------------------------------------
# PANOPTIC QUALITY
# ------------------------------------------------------------

print(
    f"PQ Proxy*      : "
    f"{float(np.mean(per_image_pq)):.4f}"
    f" ± {pq_sd:.4f}"
)


# ------------------------------------------------------------
# COMPUTATIONAL RESULTS
# ------------------------------------------------------------

print()
print(
    "COMPUTATIONAL PERFORMANCE"
)

print("-" * 80)


if gflops is not None:

    # FLOPs are deterministic for a fixed architecture
    # and fixed input size, therefore SD = 0.

    print(
        f"FLOPs          : "
        f"{gflops:.4f} GFLOPs"
        f" ± 0.0000"
    )

else:

    print(
        "FLOPs          : unavailable"
    )


print(
    f"Inference Time : "
    f"{inference_mean:.4f}"
    f" ± {inference_sd:.4f} ms/image"
)


print(
    f"FPS            : "
    f"{fps_mean:.2f}"
    f" ± {fps_sd:.2f} images/sec"
)


print(
    f"CPU Usage      : "
    f"{cpu_mean:.2f}"
    f" ± {cpu_sd:.2f} %"
)


print(
    f"RAM Usage      : "
    f"{memory_mean:.2f}"
    f" ± {memory_sd:.2f} MB"
)


if gpu_memory_mb is not None:

    print(
        f"Peak GPU Memory: "
        f"{gpu_memory_mb:.2f} MB"
    )


# ============================================================
# DESIRED TARGET VALUES
# ============================================================

print()
print("=" * 80)
print("DESIRED / TARGET VALUES")
print("=" * 80)

print()

print(
    "mIoU              : >= 0.90"
)

print(
    "Precision          : >= 0.90"
)

print(
    "Recall             : >= 0.90"
)

print(
    "F1 Score           : >= 0.90"
)

print(
    "Dice               : >= 0.90"
)

print(
    "Sensitivity        : >= 0.90"
)

print(
    "Specificity        : >= 0.95"
)

print(
    "PQ Proxy           : >= 0.80"
)

print(
    "Inference Time     : <= 50 ms/image"
)

print(
    "FPS                : >= 20 images/sec"
)

print(
    "CPU Usage          : <= 80 %"
)

print(
    "Memory Usage       : <= 2 GB"
)

print()

print(
    "NOTE: Computational targets depend on the "
    "deployment hardware and are not universal "
    "scientific standards."
)

print(
    "NOTE: PQ is a semantic-segmentation proxy "
    "because this dataset has semantic class masks, "
    "not instance IDs."
)

print()
print("=" * 80)
print("EXTENDED EVALUATION COMPLETE")
print("=" * 80)

print()
print(
    "30-image visualization directory:"
)
 
print(
    os.path.abspath(
        EXTENDED_RESULTS_DIR
    )
)

print("=" * 80)
# ============================================================
# ADD-ON: HEATMAP + PARAMETER COUNT
# ============================================================
#
# This block DOES NOT modify your Y-Net architecture or training.
#
# Outputs:
#   1. Total number of parameters
#   2. Trainable parameters
#   3. Non-trainable parameters
#   4. Approximate model size
#   5. Grad-CAM-style segmentation heatmaps
#   6. Heatmap overlays on original images
#
# ============================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# 1. PARAMETER COUNT
# ============================================================

def count_model_parameters(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable_params = (
        total_params - trainable_params
    )

    # Approximate FP32 model size
    model_size_mb = (
        total_params * 4
        / (1024 ** 2)
    )

    print()
    print("=" * 70)
    print("Y-NET MODEL PARAMETERS")
    print("=" * 70)

    print(
        f"Total Parameters       : "
        f"{total_params:,}"
    )

    print(
        f"Trainable Parameters   : "
        f"{trainable_params:,}"
    )

    print(
        f"Non-Trainable Params   : "
        f"{non_trainable_params:,}"
    )

    print(
        f"Model Size (FP32)     : "
        f"{model_size_mb:.2f} MB"
    )

    print(
        f"Total Parameters      : "
        f"{total_params / 1e6:.2f} Million"
    )

    print("=" * 70)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "size_mb": model_size_mb,
    }


parameter_results = count_model_parameters(model)


# ============================================================
# 2. HEATMAP OUTPUT DIRECTORY
# ============================================================

HEATMAP_DIR = os.path.join(
    PRED_DIR,
    "heatmaps"
)

os.makedirs(
    HEATMAP_DIR,
    exist_ok=True
)


# ============================================================
# 3. SEGMENTATION GRAD-CAM
# ============================================================
#
# For semantic segmentation there is no single class score
# like in image classification.
#
# Therefore, this heatmap uses the model's predicted classes
# and their confidence as the target signal.
#
# The gradients are collected from the selected encoder layer.
#
# ============================================================

class SegmentationGradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.backward_hook
            )
        )

    def forward_hook(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0]
        )

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        terrain_logits, _ = (
            self.model(image)
        )

        probabilities = torch.softmax(
            terrain_logits,
            dim=1
        )

        # Predicted class for every pixel
        predicted_classes = (
            probabilities.argmax(
                dim=1
            )
        )

        # Confidence of the predicted class
        predicted_confidence = (
            probabilities.gather(
                1,
                predicted_classes.unsqueeze(1)
            )
            .squeeze(1)
        )

        # Use total predicted confidence as
        # the segmentation Grad-CAM target.
        score = predicted_confidence.mean()

        score.backward(
            retain_graph=True
        )

        activations = self.activations
        gradients = self.gradients

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Global-average-pool gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep positive evidence
        cam = F.relu(cam)

        # ----------------------------------------------------
        # Resize heatmap to input resolution
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Normalize each image independently
        # ----------------------------------------------------

        cam_min = cam.flatten(
            1
        ).min(
            dim=1
        )[0].view(
            -1, 1, 1, 1
        )

        cam_max = cam.flatten(
            1
        ).max(
            dim=1
        )[0].view(
            -1, 1, 1, 1
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return (
            cam.detach(),
            predicted_classes.detach(),
            predicted_confidence.detach()
        )


# ============================================================
# 4. SELECT TARGET LAYER
# ============================================================
#
# EfficientNet-B0's final feature layer is a good choice for
# a coarse semantic heatmap.
#
# In your Y-Net:
#
# model.encoder.features
#
# contains the EfficientNet feature blocks.
#
# ============================================================

target_layer = model.encoder.features[-1]

grad_cam = SegmentationGradCAM(
    model=model,
    target_layer=target_layer
)


# ============================================================
# 5. HEATMAP GENERATION
# ============================================================

print()
print("=" * 70)
print("GENERATING SEGMENTATION HEATMAPS")
print("=" * 70)


model.eval()

heatmaps_saved = 0

MAX_HEATMAP_IMAGES = 30


for images, masks, paths in val_loader:

    if heatmaps_saved >= MAX_HEATMAP_IMAGES:
        break

    images = images.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Grad-CAM requires gradients
    # --------------------------------------------------------

    cam, predictions, confidence = (
        grad_cam.generate(images)
    )

    # --------------------------------------------------------
    # Convert normalized image back to RGB
    # --------------------------------------------------------

    original = (
        extended_denormalize(images)
        if "extended_denormalize" in globals()
        else denormalize(images)
    )

    original = original.detach().cpu()

    cam = cam.detach().cpu()

    predictions = predictions.detach().cpu()

    confidence = confidence.detach().cpu()


    # ========================================================
    # SAVE EACH IMAGE
    # ========================================================

    for i in range(
        images.size(0)
    ):

        if heatmaps_saved >= MAX_HEATMAP_IMAGES:
            break

        raw = (
            original[i]
            .permute(
                1,
                2,
                0
            )
            .numpy()
        )

        raw = np.clip(
            raw,
            0,
            1
        )

        heatmap = (
            cam[i, 0]
            .numpy()
        )


        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        base_name = Path(
            paths[i]
        ).stem

        # ----------------------------------------------------
        # 1. Pure heatmap
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 6)
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1
        )

        plt.colorbar(
            label="Activation"
        )

        plt.axis("off")

        plt.title(
            "Y-Net Segmentation Heatmap"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                HEATMAP_DIR,
                f"{heatmaps_saved + 1:02d}_{base_name}_heatmap.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


        # ----------------------------------------------------
        # 2. Heatmap overlay
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 6)
        )

        plt.imshow(
            raw
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1
        )

        plt.colorbar(
            label="Activation"
        )

        plt.axis("off")

        plt.title(
            "Y-Net Heatmap Overlay"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                HEATMAP_DIR,
                f"{heatmaps_saved + 1:02d}_{base_name}_overlay.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


        # ----------------------------------------------------
        # 3. Original + heatmap + prediction
        # ----------------------------------------------------

        prediction = (
            predictions[i]
            .numpy()
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6)
        )

        axes[0].imshow(
            raw
        )

        axes[0].set_title(
            "Original Image"
        )

        axes[0].axis("off")


        axes[1].imshow(
            raw
        )

        axes[1].imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1
        )

        axes[1].set_title(
            "Segmentation Heatmap"
        )

        axes[1].axis("off")


        axes[2].imshow(
            prediction,
            cmap="tab20",
            vmin=0,
            vmax=NUM_CLASSES - 1
        )

        axes[2].set_title(
            "Predicted Segmentation"
        )

        axes[2].axis("off")


        fig.suptitle(
            f"Y-Net | Mean Confidence: "
            f"{confidence[i].mean().item():.4f}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                HEATMAP_DIR,
                f"{heatmaps_saved + 1:02d}_{base_name}_complete.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


        heatmaps_saved += 1


# ============================================================
# 6. CLEAN UP GRAD-CAM HOOKS
# ============================================================

grad_cam.remove_hooks()


# ============================================================
# 7. FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("HEATMAP + PARAMETER ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Heatmaps generated : "
    f"{heatmaps_saved}"
)

print(
    "Heatmap directory  : "
    f"{os.path.abspath(HEATMAP_DIR)}"
)

print()
print(
    f"Total parameters   : "
    f"{parameter_results['total']:,}"
)

print(
    f"Trainable params   : "
    f"{parameter_results['trainable']:,}"
)

print(
    f"Model size         : "
    f"{parameter_results['size_mb']:.2f} MB"
)

print("=" * 70)
