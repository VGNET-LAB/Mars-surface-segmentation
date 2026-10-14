# ============================================================
# binary_YNet.py
# Lightweight Partially-Binarized Y-Net for 12-Class Martian
# Terrain Segmentation
#
# ============================================================
# WINDOWS / VS CODE EDITION
# ============================================================
# This file is designed for local execution in VS Code on Windows.
#
# USER INPUT:
#   1. Edit ZIP_PATH near the top of this file.
#
# AUTOMATIC:
#   - Dataset extraction
#   - images/masks discovery
#   - output folders
#   - checkpoint folder
#   - prediction/visualization folders
#   - GPU/CPU selection
#
# Example:
#   ZIP_PATH = r"C:\Users\gpu\Desktop\srijita\mars_dataset_10000.zip"
# ============================================================

# Based on the uploaded Y-Net pipeline:
# EfficientNet-B0 + AGCB + CSAIB + FARB + Boundary Branch
# + rare-class-aware sampling
# + Weighted CE + Class-aware Dice + Focal + Tversky
# + Boundary loss + full evaluation + SD + computational metrics
#
# Binarization strategy:
#   - EfficientNet early feature extraction remains FP.
#   - Deep EfficientNet feature blocks can be binarized.
#   - Y-Net bottleneck remains binary.
#   - Decoder is fully FP (dec1-dec5), including its refinement and attention convolutions.
#   - Final terrain and boundary heads remain FP.
#
# IMPORTANT:
#   "0.85 accuracy" is treated as a target, not a guarantee.
#   For segmentation, mIoU is reported separately from pixel accuracy.
#   The binary model is intended to target approximately >=0.85 mIoU
#   while substantially reducing effective binary weight storage.
#
# ============================================================

import os
import re
import gc
import math
import time
import random
import zipfile
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 42

# ============================================================
# WINDOWS / VS CODE PATH CONFIGURATION
# ============================================================
# EDIT ONLY THIS PATH.
# Example:
# ZIP_PATH = r"E:\val\mars_dataset_10000_1134.zip"
#
# Do NOT use:
# ZIP_PATH = "C:\Users\gpu\Desktop\srijita\..."
# because backslashes can be interpreted as escape characters.
# ============================================================

ZIP_PATH = r"E:\val\mars_dataset_10000_1134.zip"

# The remaining paths are generated automatically relative to
# the folder containing this Python script.
SCRIPT_DIR = Path(__file__).resolve().parent

# Dataset extraction directory.
EXTRACT_DIR = SCRIPT_DIR / "mars_dataset_10000"

# Output directories.
OUTPUT_DIR = SCRIPT_DIR / "binary_YNet_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR = OUTPUT_DIR / "predictions"
EXTENDED_RESULTS_DIR = PRED_DIR / "extended_evaluation"
HEATMAP_DIR = PRED_DIR / "heatmaps"

# Converted single-channel class-label masks.
# These are generated from the original RGB masks and are then used
# by the dataset/training pipeline.
LABEL_DIR = OUTPUT_DIR / "labels"

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_binary_y_net_mars.pth"

NUM_CLASSES = 12
IMG_SIZE = 384

BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 180

# Slightly lower LR than the original FP model because binary
# layers use a straight-through estimator.
LR = 1e-4
ENCODER_LR = 2e-5

WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-5
PATIENCE = 25

AMP = True
GRAD_ACCUM_STEPS = 2

# False is recommended for actual lightweight inference/FPS
# benchmarking. TTA remains available as an optional setting.
USE_TTA = False

VAL_RATIO = 0.20

USE_PRETRAINED = True
USE_RARE_CLASS_SAMPLER = False

RARE_PIXEL_THRESHOLD = 0.025
MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE = 500

MAX_VISUALIZATIONS = 30
MAX_HEATMAP_IMAGES = 30

# ------------------------------------------------------------
# Binary model controls
# ------------------------------------------------------------

BINARY_EPS = 1e-8

# Binarize the deep EfficientNet feature stages.
# EfficientNet feature indices:
# 0 stem
# 1-7 MBConv/deep feature stages
# 8 final 1x1 feature block
BINARY_EFFICIENTNET_INDICES = (5, 6, 7, 8)

# Keep the first two feature stages FP for stability.
BINARY_DECODER = False
BINARY_ATTENTION = False

# Decoder is kept full precision to preserve spatial reconstruction and boundary detail.
# Lightweight channel configuration.
BOTTLENECK_CH = 128
DEC1_CH = 192
DEC2_CH = 128
DEC3_CH = 64
DEC4_CH = 32
FINAL_CH = 24

# Accuracy-oriented training profile: FP skip-attention + wider FP decoder.
# The deep encoder and bottleneck binary blocks remain binary.
# Experiment C: FP shallow encoder stage + binary deep encoder + FP attention + FP decoder.
# Binary contribution is retained in EfficientNet stages 5-8 and the bottleneck.


# ============================================================
# 12 RGB MASK COLORS FROM THE ORIGINAL PIPELINE
# ============================================================

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

CLASS_NAMES = [
    f"Class_{i}" for i in range(NUM_CLASSES)
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# 1. REPRODUCIBILITY / DEVICE
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type != "cuda":
    print(
        "WARNING: CUDA GPU was not detected. "
        "Binary training will run on CPU."
    )


# ============================================================
# 2. DATASET EXTRACTION
# ============================================================

def find_zip():
    """Return the user-specified dataset ZIP."""
    zip_file = Path(ZIP_PATH).expanduser()

    if not zip_file.exists():
        raise FileNotFoundError(
            "\nDataset ZIP was not found.\n\n"
            f"Expected path:\n{zip_file}\n\n"
            "Please edit ZIP_PATH at the top of this file and "
            "run the script again."
        )

    if zip_file.suffix.lower() != ".zip":
        raise ValueError(
            f"ZIP_PATH must point to a .zip file:\n{zip_file}"
        )

    return zip_file.resolve()


def find_dataset_dirs(root):
    """
    Locate image and mask directories anywhere under root.

    Supported layouts include:
        root/images + root/masks
        root/images + root/label_masks
        root/<nested>/images + root/<nested>/masks
    """
    root = Path(root)

    image_candidates = []
    mask_candidates = []

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        name = p.name.lower()

        if name == "images":
            image_candidates.append(p)

        if name in {"masks", "label_masks", "labels", "label"}:
            mask_candidates.append(p)

    # Prefer directories containing PNG files.
    image_candidates = [
        p for p in image_candidates
        if any(p.glob("*.png"))
    ]

    mask_candidates = [
        p for p in mask_candidates
        if any(p.glob("*.png"))
    ]

    if not image_candidates:
        raise RuntimeError(
            "\nCould not find an 'images' directory containing PNG files.\n"
            f"Search root: {root}\n\n"
            "Check the ZIP contents and make sure the dataset contains "
            "an images folder."
        )

    if not mask_candidates:
        raise RuntimeError(
            "\nCould not find a 'masks', 'label_masks', 'labels', "
            "or 'label' directory containing PNG files.\n"
            f"Search root: {root}\n\n"
            "Check the ZIP contents and make sure the dataset contains "
            "a mask/label folder."
        )

    # If several candidates exist, choose the pair with the largest
    # number of files. This is safer for ZIPs containing extra folders.
    image_dir = max(
        image_candidates,
        key=lambda p: len(list(p.glob("*.png")))
    )
    mask_dir = max(
        mask_candidates,
        key=lambda p: len(list(p.glob("*.png")))
    )

    return image_dir, mask_dir


zip_file = find_zip()

# Create output folders before training.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
EXTENDED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Check whether the dataset has already been extracted.
# If extraction exists but images/masks cannot be found, re-extract.
dataset_ready = False

if EXTRACT_DIR.exists():
    try:
        _existing_image_dir, _existing_mask_dir = find_dataset_dirs(
            EXTRACT_DIR
        )
        dataset_ready = True
        print("Existing extracted dataset found.")
    except Exception:
        dataset_ready = False

if not dataset_ready:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("EXTRACTING MARS DATASET")
    print("=" * 80)
    print("ZIP:", zip_file)
    print("Destination:", EXTRACT_DIR)
    print()

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(EXTRACT_DIR)

    print("Extraction completed.")

IMAGE_DIR, MASK_DIR = find_dataset_dirs(EXTRACT_DIR)

print()
print("=" * 80)
print("DATASET PATHS")
print("=" * 80)
print("ZIP file     :", zip_file)
print("Extracted at :", EXTRACT_DIR)
print("Image dir    :", IMAGE_DIR)
print("Mask dir     :", MASK_DIR)
print("Output dir   :", OUTPUT_DIR)
print("Label dir    :", LABEL_DIR)
print("Checkpoint   :", CHECKPOINT_PATH)
print("=" * 80)
print()

# ============================================================
# 3. IMAGE/MASK PAIRING
# ============================================================

def build_pairs(image_dir, mask_dir):
    image_files = sorted(
        Path(image_dir).glob("*.png")
    )

    mask_files = sorted(
        Path(mask_dir).glob("*.png")
    )

    mask_by_key = {}

    for m in mask_files:
        key = m.stem.replace("mask_", "", 1)
        mask_by_key[key] = m

    pairs = []

    for im in image_files:
        key = im.stem.replace("img_", "", 1)

        if key in mask_by_key:
            pairs.append((im, mask_by_key[key]))

    return pairs


pairs = build_pairs(
    IMAGE_DIR,
    MASK_DIR
)

if len(pairs) == 0:
    raise RuntimeError(
        "No image-mask pairs were found."
    )

print(f"Found {len(pairs)} image-mask pairs.")


# ============================================================
# 4. RGB MASK -> CLASS IDS
# ============================================================

palette = np.asarray(
    PALETTE_RGB,
    dtype=np.int16
)


def rgb_mask_to_labels(mask_rgb):
    arr = np.asarray(
        mask_rgb.convert("RGB"),
        dtype=np.int16
    )

    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)

    labels = np.empty(
        flat.shape[0],
        dtype=np.uint8
    )

    chunk = 250_000

    for start in range(
        0,
        len(flat),
        chunk
    ):
        end = min(
            start + chunk,
            len(flat)
        )

        x = flat[start:end].astype(
            np.int32
        )

        distances = (
            (
                x[:, None, :] -
                palette[None, :, :]
            ) ** 2
        ).sum(axis=2)

        labels[start:end] = np.argmin(
            distances,
            axis=1
        ).astype(np.uint8)

    return labels.reshape(h, w)


def convert_masks_to_labels(pairs):
    converted = []

    for idx, (
        im_path,
        mask_path
    ) in enumerate(pairs):

        out_path = (
            Path(LABEL_DIR) /
            f"{mask_path.stem}.png"
        )

        if not out_path.exists():
            rgb = Image.open(
                mask_path
            ).convert("RGB")

            labels = rgb_mask_to_labels(
                rgb
            )

            Image.fromarray(
                labels,
                mode="L"
            ).save(out_path)

        converted.append(
            (im_path, out_path)
        )

        if (
            (idx + 1) % 100 == 0
            or idx == 0
        ):
            print(
                f"Converted masks: "
                f"{idx + 1}/{len(pairs)}"
            )

    return converted


pairs = convert_masks_to_labels(
    pairs
)


# ============================================================
# 5. DATASET STATISTICS
# ============================================================

def mask_stats(mask_path):
    mask = np.asarray(
        Image.open(mask_path),
        dtype=np.uint8
    )

    return np.bincount(
        mask.reshape(-1),
        minlength=NUM_CLASSES
    )


all_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.int64
)

image_class_counts = []

for _, mask_path in pairs:
    counts = mask_stats(mask_path)

    all_counts += counts
    image_class_counts.append(counts)


total_pixels = all_counts.sum()

freq = (
    all_counts /
    max(total_pixels, 1)
)

print("\n========== DATASET CLASS DISTRIBUTION ==========")

for c in range(NUM_CLASSES):
    image_presence = sum(
        counts[c] > 0
        for counts in image_class_counts
    )

    print(
        f"{c:2d} | "
        f"{CLASS_NAMES[c]:10s} | "
        f"pixels={all_counts[c]:12d} | "
        f"{freq[c] * 100:7.3f}% | "
        f"images={image_presence:4d}"
    )


rare_classes = np.where(
    freq < RARE_PIXEL_THRESHOLD
)[0].tolist()

print("Rare classes:", rare_classes)


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

indices = np.arange(
    len(pairs)
)

rare_presence = np.array([
    any(
        counts[c] >=
        MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE
        for c in rare_classes
    )
    if rare_classes
    else False
    for counts in image_class_counts
])

rare_idx = indices[rare_presence]
normal_idx = indices[~rare_presence]

rng.shuffle(rare_idx)
rng.shuffle(normal_idx)


def split_group(
    idx_array,
    val_ratio
):
    if len(idx_array) == 0:
        return idx_array, idx_array

    n_val = max(
        1,
        int(
            round(
                len(idx_array) *
                val_ratio
            )
        )
    )

    return (
        idx_array[n_val:],
        idx_array[:n_val]
    )


train_r, val_r = split_group(
    rare_idx,
    VAL_RATIO
)

train_n, val_n = split_group(
    normal_idx,
    VAL_RATIO
)

train_indices = np.concatenate(
    [train_r, train_n]
)

val_indices = np.concatenate(
    [val_r, val_n]
)

rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_pairs = [
    pairs[i]
    for i in train_indices
]

val_pairs = [
    pairs[i]
    for i in val_indices
]

print("\nTrain pairs:", len(train_pairs))
print("Val pairs  :", len(val_pairs))


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

train_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

train_image_counts = []

for _, mask_path in train_pairs:
    counts = mask_stats(mask_path)

    train_counts += counts
    train_image_counts.append(counts)


train_freq = (
    train_counts /
    max(train_counts.sum(), 1)
)

class_weights = (
    1.0 /
    np.sqrt(
        np.maximum(
            train_freq,
            1e-8
        )
    )
)

class_weights /= (
    class_weights.mean()
)

class_weights = np.clip(
    class_weights,
    0.50,
    3.0
)

print("\nClass weights:")

for c, w in enumerate(
    class_weights
):
    print(
        f"{c:2d} "
        f"{CLASS_NAMES[c]:10s}: "
        f"{w:.3f}"
    )


# ============================================================
# 8. RARE-CLASS-AWARE SAMPLER
# ============================================================

def make_sampler_weights(
    image_counts
):
    weights = []

    for counts in image_counts:
        score = 1.0

        for c in rare_classes:
            if (
                counts[c] >=
                MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE
            ):
                score += (
                    1.5 *
                    float(class_weights[c])
                )

        weights.append(score)

    return torch.as_tensor(
        weights,
        dtype=torch.double
    )


sampler = None
shuffle = True

if USE_RARE_CLASS_SAMPLER:
    sampler_weights = (
        make_sampler_weights(
            train_image_counts
        )
    )

    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(
            sampler_weights
        ),
        replacement=True
    )

    shuffle = False


# ============================================================
# 9. PAIRED AUGMENTATIONS
# ============================================================

def paired_train_transform(
    image,
    mask
):
    w, h = image.size

    scale = random.uniform(
        0.85,
        1.0
    )

    crop_w = max(
        1,
        min(
            w,
            int(w * scale)
        )
    )

    crop_h = max(
        1,
        min(
            h,
            int(h * scale)
        )
    )

    left = random.randint(
        0,
        max(
            0,
            w - crop_w
        )
    )

    top = random.randint(
        0,
        max(
            0,
            h - crop_h
        )
    )

    image = TF.crop(
        image,
        top,
        left,
        crop_h,
        crop_w
    )

    mask = TF.crop(
        mask,
        top,
        left,
        crop_h,
        crop_w
    )

    image = TF.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        interpolation=(
            InterpolationMode.BILINEAR
        )
    )

    mask = TF.resize(
        mask,
        [IMG_SIZE, IMG_SIZE],
        interpolation=(
            InterpolationMode.NEAREST
        )
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
                image,
                angle,
                interpolation=(
                    InterpolationMode.BILINEAR
                )
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=(
                    InterpolationMode.NEAREST
                )
            )

    if random.random() < 0.7:
        image = TF.adjust_brightness(
            image,
            random.uniform(
                0.88,
                1.12
            )
        )

        image = TF.adjust_contrast(
            image,
            random.uniform(
                0.88,
                1.15
            )
        )

    if random.random() < 0.05:
        image = TF.gaussian_blur(
            image,
            kernel_size=3,
            sigma=(0.1, 0.8)
        )

    return image, mask


def paired_val_transform(
    image,
    mask
):
    image = TF.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        interpolation=(
            InterpolationMode.BILINEAR
        )
    )

    mask = TF.resize(
        mask,
        [IMG_SIZE, IMG_SIZE],
        interpolation=(
            InterpolationMode.NEAREST
        )
    )

    return image, mask


class MarsDataset(Dataset):

    def __init__(
        self,
        pairs,
        train=True
    ):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = (
            self.pairs[idx]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        if self.train:
            image, mask = (
                paired_train_transform(
                    image,
                    mask
                )
            )
        else:
            image, mask = (
                paired_val_transform(
                    image,
                    mask
                )
            )

        image = TF.to_tensor(
            image
        )

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

        return (
            image,
            mask,
            str(image_path)
        )


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
    pin_memory=(
        DEVICE.type == "cuda"
    ),
    persistent_workers=(
        NUM_WORKERS > 0
    )
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
    persistent_workers=(
        NUM_WORKERS > 0
    )
)


# ============================================================
# 10. BINARY STE
# ============================================================

class BinarySignSTE(torch.autograd.Function):
    """
    Straight-through estimator for sign activation.

    Forward:
        sign(x)

    Backward:
        gradient is passed through inside |x| <= 1 and clipped
        outside that range.
    """

    @staticmethod
    def forward(
        ctx,
        x
    ):
        ctx.save_for_backward(x)

        return torch.where(
            x >= 0,
            torch.ones_like(x),
            -torch.ones_like(x)
        )

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):
        (x,) = ctx.saved_tensors

        grad = grad_output.clone()

        mask = (
            x.abs() <= 1.0
        ).to(grad.dtype)

        return grad * mask


def binary_sign(x):
    return BinarySignSTE.apply(x)


class BinaryConv2d(nn.Module):
    """
    Binary-weight convolution.

    The underlying real-valued weight remains trainable.
    During forward propagation it is binarized with STE.

    A per-output-channel scale factor is used:

        alpha = mean(abs(weight))

        W_binary = sign(W) * alpha

    This is much more stable than using raw +/-1 weights.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False
    ):
        super().__init__()

        if isinstance(
            kernel_size,
            int
        ):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if isinstance(
            stride,
            int
        ):
            sh = sw = stride
        else:
            sh, sw = stride

        if isinstance(
            padding,
            int
        ):
            ph = pw = padding
        else:
            ph, pw = padding

        if isinstance(
            dilation,
            int
        ):
            dh = dw = dilation
        else:
            dh, dw = dilation

        self.in_channels = (
            in_channels
        )
        self.out_channels = (
            out_channels
        )
        self.kernel_size = (
            kh,
            kw
        )
        self.stride = (
            sh,
            sw
        )
        self.padding = (
            ph,
            pw
        )
        self.dilation = (
            dh,
            dw
        )
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kh,
                kw
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    out_channels
                )
            )
        else:
            self.bias = None

        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

    def forward(self, x):
        scale = (
            self.weight
            .abs()
            .mean(
                dim=(1, 2, 3),
                keepdim=True
            )
        )

        binary_weight = (
            binary_sign(
                self.weight
            ) *
            scale.clamp_min(
                BINARY_EPS
            )
        )

        return F.conv2d(
            x,
            binary_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class BinaryConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        p=1,
        groups=1
    ):
        super().__init__()

        self.conv = BinaryConv2d(
            in_ch,
            out_ch,
            k,
            padding=p,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            out_ch
        )

        self.act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        p=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                k,
                padding=p,
                bias=False
            ),
            nn.BatchNorm2d(
                out_ch
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 11. AGCB
# ============================================================

class AGCB(nn.Module):
    """
    Adaptive Gated Convolutional Block.

    Binary 3x3 feature convolutions are used when
    BINARY_ATTENTION=True.
    """

    def __init__(
        self,
        channels,
        reduction=8,
        binary=True
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        Conv = (
            BinaryConv2d
            if binary
            else nn.Conv2d
        )

        self.conv1 = Conv(
            channels,
            channels,
            3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = Conv(
            channels,
            channels,
            3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        # 1x1 gates remain FP because they control the feature
        # selection and are relatively small.
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.out_act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(
            y,
            inplace=True
        )

        y = self.conv2(y)
        y = self.bn2(y)

        gate = self.gate(y)

        y = y * gate

        return self.out_act(
            x + y
        )


# ============================================================
# 12. CSAIB
# ============================================================

class CSAIB(nn.Module):
    """
    Channel + Spatial Attention Integration Block.

    The original 7x7 spatial convolution is reduced to 3x3
    for the lightweight binary model.
    """

    def __init__(
        self,
        channels,
        reduction=8,
        binary=True
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
                bias=False
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=False
            )
        )

        SpatialConv = (
            BinaryConv2d
            if binary
            else nn.Conv2d
        )

        self.spatial = SpatialConv(
            2,
            1,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(
            x,
            1
        )

        mx = F.adaptive_max_pool2d(
            x,
            1
        )

        channel_attn = torch.sigmoid(
            self.mlp(avg) +
            self.mlp(mx)
        )

        x = (
            x *
            channel_attn
        )

        avg_sp = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_sp, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        spatial_input = torch.cat(
            [avg_sp, max_sp],
            dim=1
        )

        spatial_attn = torch.sigmoid(
            self.spatial(
                spatial_input
            )
        )

        return (
            x *
            spatial_attn
        )


# ============================================================
# 13. FARB
# ============================================================

class FARB(nn.Module):
    """
    Fusion Attention Refinement Block.

    1x1 fusion is FP; 3x3 refinement is binary.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        binary=True
    ):
        super().__init__()

        Conv3 = (
            BinaryConv2d
            if binary
            else nn.Conv2d
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False
            ),
            nn.BatchNorm2d(
                out_ch
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.fuse2 = nn.Sequential(
            Conv3(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                out_ch
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.attn = CSAIB(
            out_ch,
            binary=binary
        )

        self.residual = (
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)

        y = self.fuse1(x)
        y = self.fuse2(y)
        y = self.attn(y)

        return F.relu(
            y + identity,
            inplace=True
        )


# ============================================================
# 14. EFFICIENTNET ENCODER
# ============================================================

class EfficientNetEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        try:
            backbone = efficientnet_b0(
                weights=weights
            )
        except Exception as exc:
            print(
                "WARNING: Could not load "
                "pretrained EfficientNet-B0."
            )
            print("Reason:", exc)
            print(
                "Using random initialization."
            )

            backbone = efficientnet_b0(
                weights=None
            )

        self.features = (
            backbone.features
        )

        # Replace selected deep convolutional layers
        # recursively. This preserves EfficientNet's MBConv
        # structure, including depthwise groups.
        if BINARY_EFFICIENTNET_INDICES:
            for idx in BINARY_EFFICIENTNET_INDICES:
                if idx < len(
                    self.features
                ):
                    self.features[idx] = (
                        self._binarize_module(
                            self.features[idx]
                        )
                    )

    def _binarize_module(
        self,
        module
    ):
        for name, child in list(
            module.named_children()
        ):
            if isinstance(
                child,
                nn.Conv2d
            ):
                # Keep 1x1 pointwise convolutions FP.
                # Binarize spatial 3x3/5x5 convolutions,
                # including depthwise convolutions.
                k = child.kernel_size

                if (
                    k[0] > 1
                    or k[1] > 1
                ):
                    new_conv = BinaryConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=(
                            child.bias is not None
                        )
                    )

                    with torch.no_grad():
                        new_conv.weight.copy_(
                            child.weight
                        )

                        if (
                            child.bias is not None
                            and new_conv.bias is not None
                        ):
                            new_conv.bias.copy_(
                                child.bias
                            )

                    setattr(
                        module,
                        name,
                        new_conv
                    )

            else:
                self._binarize_module(
                    child
                )

        return module

    def forward(self, x):
        skips = []

        for i, layer in enumerate(
            self.features
        ):
            x = layer(x)

            # Same skip locations as the uploaded Y-Net.
            if i in (
                0,
                2,
                3,
                4
            ):
                skips.append(x)

        return skips, x


# ============================================================
# 15. DECODER
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch
    ):
        super().__init__()

        self.farb = FARB(
            in_ch + skip_ch,
            out_ch,
            binary=BINARY_DECODER
        )

    def forward(
        self,
        x,
        skip
    ):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.farb(x)


# ============================================================
# 16. BINARY Y-NET
# ============================================================

class BinaryYNet(nn.Module):

    def __init__(
        self,
        num_classes=12,
        pretrained=True
    ):
        super().__init__()

        self.encoder = (
            EfficientNetEncoder(
                pretrained=pretrained
            )
        )

        # Feature enhancement blocks.
        self.agcb0 = AGCB(
            32,
            binary=BINARY_ATTENTION
        )
        self.csai0 = CSAIB(
            32,
            binary=BINARY_ATTENTION
        )

        self.agcb1 = AGCB(
            24,
            binary=BINARY_ATTENTION
        )
        self.csai1 = CSAIB(
            24,
            binary=BINARY_ATTENTION
        )

        self.agcb2 = AGCB(
            40,
            binary=BINARY_ATTENTION
        )
        self.csai2 = CSAIB(
            40,
            binary=BINARY_ATTENTION
        )

        self.agcb3 = AGCB(
            80,
            binary=BINARY_ATTENTION
        )
        self.csai3 = CSAIB(
            80,
            binary=BINARY_ATTENTION
        )

        # ----------------------------------------------------
        # Lightweight bottleneck
        # ----------------------------------------------------

        self.bottleneck_reduce = (
            ConvBNReLU(
                1280,
                BOTTLENECK_CH,
                k=1,
                p=0
            )
        )

        self.bottleneck_agcb = AGCB(
            BOTTLENECK_CH,
            binary=True
        )

        self.bottleneck_csai = CSAIB(
            BOTTLENECK_CH,
            binary=True
        )

        self.bottleneck_farb = FARB(
            BOTTLENECK_CH,
            BOTTLENECK_CH,
            binary=True
        )

        # ----------------------------------------------------
        # Lightweight decoder
        # ----------------------------------------------------

        self.dec1 = DecoderBlock(
            BOTTLENECK_CH,
            80,
            DEC1_CH
        )

        self.dec2 = DecoderBlock(
            DEC1_CH,
            40,
            DEC2_CH
        )

        self.dec3 = DecoderBlock(
            DEC2_CH,
            24,
            DEC3_CH
        )

        self.dec4 = DecoderBlock(
            DEC3_CH,
            32,
            DEC4_CH
        )

        # Final decoder refinement is FP.
        # The entire decoder (dec1-dec5) remains full precision.
        self.dec5 = nn.Sequential(
            ConvBNReLU(
                DEC4_CH,
                FINAL_CH
            ),
            ConvBNReLU(
                FINAL_CH,
                FINAL_CH
            )
        )

        # ----------------------------------------------------
        # Final Y-Net heads remain FP.
        # ----------------------------------------------------

        self.terrain_head = nn.Conv2d(
            FINAL_CH,
            num_classes,
            1
        )

        self.boundary_head = nn.Conv2d(
            FINAL_CH,
            1,
            1
        )

    def forward(self, x):
        skips, bottleneck = (
            self.encoder(x)
        )

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

        x = self.dec1(
            x,
            s3
        )

        x = self.dec2(
            x,
            s2
        )

        x = self.dec3(
            x,
            s1
        )

        x = self.dec4(
            x,
            s0
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.dec5(x)

        terrain_logits = (
            self.terrain_head(x)
        )

        boundary_logits = (
            self.boundary_head(x)
        )

        return (
            terrain_logits,
            boundary_logits
        )


model = BinaryYNet(
    num_classes=NUM_CLASSES,
    pretrained=USE_PRETRAINED
).to(DEVICE)


# ============================================================
# 17. BINARY MODEL STATISTICS
# ============================================================

def is_binary_parameter_name(
    name,
    module
):
    return isinstance(
        module,
        BinaryConv2d
    )


def binary_parameter_statistics(
    model
):
    total_params = 0
    binary_params = 0
    trainable_params = 0
    fp_params = 0

    binary_module_count = 0

    for module_name, module in (
        model.named_modules()
    ):
        if isinstance(
            module,
            BinaryConv2d
        ):
            binary_module_count += 1
            binary_params += (
                module.weight.numel()
            )

    for p in model.parameters():
        n = p.numel()

        total_params += n

        if p.requires_grad:
            trainable_params += n

    fp_params = (
        total_params -
        binary_params
    )

    # Theoretical storage:
    # binary weights -> 1 bit
    # FP weights -> 32 bits
    theoretical_bits = (
        binary_params * 1 +
        fp_params * 32
    )

    fp32_bits = (
        total_params * 32
    )

    compression = (
        fp32_bits /
        max(theoretical_bits, 1)
    )

    return {
        "total_params": total_params,
        "binary_params": binary_params,
        "fp_params": fp_params,
        "trainable_params": trainable_params,
        "binary_ratio": (
            binary_params /
            max(total_params, 1)
        ),
        "fp32_size_mb": (
            fp32_bits /
            8 /
            1024 /
            1024
        ),
        "theoretical_binary_size_mb": (
            theoretical_bits /
            8 /
            1024 /
            1024
        ),
        "theoretical_compression": compression,
        "binary_module_count": (
            binary_module_count
        )
    }


binary_stats = (
    binary_parameter_statistics(
        model
    )
)

print("\n" + "=" * 70)
print("BINARY Y-NET MODEL SUMMARY")
print("=" * 70)

print(
    f"Total parameters           : "
    f"{binary_stats['total_params']:,}"
)

print(
    f"Binary-weight parameters   : "
    f"{binary_stats['binary_params']:,}"
)

print(
    f"FP parameters              : "
    f"{binary_stats['fp_params']:,}"
)

print(
    f"Binary parameter ratio     : "
    f"{binary_stats['binary_ratio'] * 100:.2f}%"
)

print(
    f"Binary convolution layers  : "
    f"{binary_stats['binary_module_count']}"
)

print(
    f"FP32 parameter storage     : "
    f"{binary_stats['fp32_size_mb']:.2f} MB"
)

print(
    f"Theoretical 1-bit storage  : "
    f"{binary_stats['theoretical_binary_size_mb']:.2f} MB"
)

print(
    f"Theoretical compression    : "
    f"{binary_stats['theoretical_compression']:.2f}x"
)

print("=" * 70)


# ============================================================
# 18. BOUNDARY TARGET
# ============================================================

def make_boundary_target(
    mask
):
    """
    Boundary = pixel whose label differs from at least one
    4-connected neighbour.
    """

    m = mask.unsqueeze(1)

    left = F.pad(
        m[:, :, :, :-1],
        (1, 0, 0, 0),
        value=255
    )

    right = F.pad(
        m[:, :, :, 1:],
        (0, 1, 0, 0),
        value=255
    )

    up = F.pad(
        m[:, :, :-1, :],
        (0, 0, 1, 0),
        value=255
    )

    down = F.pad(
        m[:, :, 1:, :],
        (0, 0, 0, 1),
        value=255
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
# 19. LOSSES
# ============================================================

class WeightedDiceLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        smooth=1.0
    ):
        super().__init__()

        self.register_buffer(
            "class_weights",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

        self.smooth = smooth

    def forward(
        self,
        logits,
        target
    ):
        probs = torch.softmax(
            logits,
            dim=1
        )

        one_hot = F.one_hot(
            target.clamp(
                0,
                NUM_CLASSES - 1
            ),
            NUM_CLASSES
        ).permute(
            0,
            3,
            1,
            2
        ).float()

        dims = (
            0,
            2,
            3
        )

        intersection = (
            probs * one_hot
        ).sum(dims)

        denominator = (
            probs.sum(dims) +
            one_hot.sum(dims)
        )

        dice = (
            2.0 * intersection +
            self.smooth
        ) / (
            denominator +
            self.smooth
        )

        weights = (
            self.class_weights /
            self.class_weights.mean()
        )

        return (
            1.0 -
            (
                dice * weights
            ).sum() /
            weights.sum()
        )


class FocalLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        gamma=2.0
    ):
        super().__init__()

        self.gamma = gamma

        self.register_buffer(
            "weight",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

    def forward(
        self,
        logits,
        target
    ):
        logp = F.log_softmax(
            logits,
            dim=1
        )

        logpt = logp.gather(
            1,
            target.unsqueeze(1)
        ).squeeze(1)

        pt = logpt.exp()

        focal = (
            -(1.0 - pt).pow(
                self.gamma
            ) *
            logpt
        )

        alpha = self.weight[target]

        return (
            alpha * focal
        ).mean()


class TverskyLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        alpha=0.30,
        beta=0.70,
        smooth=1.0
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        self.register_buffer(
            "class_weights",
            torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

    def forward(
        self,
        logits,
        target
    ):
        probs = torch.softmax(
            logits,
            dim=1
        )

        one_hot = F.one_hot(
            target.clamp(
                0,
                NUM_CLASSES - 1
            ),
            NUM_CLASSES
        ).permute(
            0,
            3,
            1,
            2
        ).float()

        dims = (
            0,
            2,
            3
        )

        tp = (
            probs * one_hot
        ).sum(dims)

        fp = (
            probs *
            (1.0 - one_hot)
        ).sum(dims)

        fn = (
            (1.0 - probs) *
            one_hot
        ).sum(dims)

        score = (
            tp + self.smooth
        ) / (
            tp +
            self.alpha * fp +
            self.beta * fn +
            self.smooth
        )

        weights = (
            self.class_weights /
            self.class_weights.mean()
        )

        return (
            1.0 -
            (
                score * weights
            ).sum() /
            weights.sum()
        )


class BinaryDiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        target
    ):
        probs = torch.sigmoid(
            logits
        )

        p = probs.reshape(-1)
        t = target.float().reshape(-1)

        intersection = (
            p * t
        ).sum()

        return (
            1.0 -
            (
                2.0 *
                intersection +
                self.smooth
            ) /
            (
                p.sum() +
                t.sum() +
                self.smooth
            )
        )


class YNetLoss(nn.Module):

    def __init__(
        self,
        class_weights
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights,
                dtype=torch.float32
            ),
            label_smoothing=0.0
        )

        self.dice = (
            WeightedDiceLoss(
                class_weights
            )
        )

        self.focal = FocalLoss(
            class_weights,
            gamma=2.0
        )

        self.tversky = TverskyLoss(
            class_weights,
            alpha=0.30,
            beta=0.70
        )

        self.boundary_bce = (
            nn.BCEWithLogitsLoss()
        )

        self.boundary_dice = (
            BinaryDiceLoss()
        )

    def forward(
        self,
        terrain_logits,
        boundary_logits,
        target
    ):
        boundary_target = (
            make_boundary_target(
                target
            )
        )

        ce_loss = self.ce(
            terrain_logits,
            target
        )

        dice_loss = self.dice(
            terrain_logits,
            target
        )

        focal_loss = self.focal(
            terrain_logits,
            target
        )

        tversky_loss = self.tversky(
            terrain_logits,
            target
        )

        bce = self.boundary_bce(
            boundary_logits,
            boundary_target
        )

        bdice = self.boundary_dice(
            boundary_logits,
            boundary_target
        )

        boundary_loss = (
            0.5 * bce +
            0.5 * bdice
        )

        # Experiment C loss: emphasize overlap and recall while
        # retaining CE for stable multi-class optimization and a
        # small boundary term for spatial detail.
        total = (
            1.00 * ce_loss +
            1.00 * dice_loss +
            0.15 * focal_loss +
            0.35 * tversky_loss +
            0.15 * boundary_loss
        )

        return total


criterion = YNetLoss(
    class_weights
).to(DEVICE)


# ============================================================
# 20. CONFUSION MATRIX + METRICS
# ============================================================

@torch.no_grad()
def update_confusion_matrix(
    confmat,
    pred,
    target,
    num_classes
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
        minlength=(
            num_classes ** 2
        )
    )

    confmat += binc.reshape(
        num_classes,
        num_classes
    )


def metrics_from_confmat(
    confmat
):
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

    denom_iou = (
        tp + fp + fn
    )

    denom_dice = (
        2 * tp + fp + fn
    )

    denom_prec = (
        tp + fp
    )

    denom_rec = (
        tp + fn
    )

    iou = np.divide(
        tp,
        denom_iou,
        out=np.zeros_like(tp),
        where=denom_iou > 0
    )

    dice = np.divide(
        2 * tp,
        denom_dice,
        out=np.zeros_like(tp),
        where=denom_dice > 0
    )

    precision = np.divide(
        tp,
        denom_prec,
        out=np.zeros_like(tp),
        where=denom_prec > 0
    )

    recall = np.divide(
        tp,
        denom_rec,
        out=np.zeros_like(tp),
        where=denom_rec > 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(
            precision + recall
        ) > 0
    )

    pixel_acc = (
        tp.sum() /
        max(
            confmat.sum(),
            1
        )
    )

    specificity_den = (
        confmat.sum() -
        (tp + fp + fn)
    ) + fp

    # Equivalent one-vs-rest TN + FP denominator.
    tn = (
        confmat.sum() -
        tp -
        fp -
        fn
    )

    specificity = np.divide(
        tn,
        tn + fp,
        out=np.zeros_like(tp),
        where=(
            tn + fp
        ) > 0
    )

    valid_specificity = (
        (tn + fp) > 0
    )

    specificity_mean = (
        float(
            specificity[
                valid_specificity
            ].mean()
        )
        if np.any(
            valid_specificity
        )
        else 0.0
    )

    # Semantic-segmentation PQ proxy.
    # For semantic masks, each class is treated as a region:
    # PQ_proxy = mean( IoU * Dice ).
    pq_proxy_per_class = (
        iou * dice
    )

    pq_proxy = float(
        np.mean(
            pq_proxy_per_class
        )
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "specificity_mean": (
            specificity_mean
        ),
        "pixel_acc": pixel_acc,
        "miou_all": float(
            iou.mean()
        ),
        "mdice_all": float(
            dice.mean()
        ),
        "miou_fg": float(
            iou[1:].mean()
        ),
        "mdice_fg": float(
            dice[1:].mean()
        ),
        "rare_miou": (
            float(
                np.mean(
                    iou[rare_classes]
                )
            )
            if rare_classes
            else 0.0
        ),
        "pq_proxy": pq_proxy
    }


# ============================================================
# 21. TTA
# ============================================================

@torch.no_grad()
def predict_with_tta(
    model,
    images
):
    variants = [
        (images, None),
        (
            torch.flip(
                images,
                dims=[3]
            ),
            "h"
        ),
        (
            torch.flip(
                images,
                dims=[2]
            ),
            "v"
        ),
        (
            torch.rot90(
                images,
                2,
                dims=[2, 3]
            ),
            "r"
        )
    ]

    logits_sum = None

    for aug, kind in variants:
        terrain_logits, _ = (
            model(aug)
        )

        if kind == "h":
            terrain_logits = torch.flip(
                terrain_logits,
                dims=[3]
            )

        elif kind == "v":
            terrain_logits = torch.flip(
                terrain_logits,
                dims=[2]
            )

        elif kind == "r":
            terrain_logits = torch.rot90(
                terrain_logits,
                2,
                dims=[2, 3]
            )

        logits_sum = (
            terrain_logits
            if logits_sum is None
            else (
                logits_sum +
                terrain_logits
            )
        )

    return (
        logits_sum /
        len(variants)
    )


# ============================================================
# 22. VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    use_tta=False
):
    model.eval()

    running_loss = 0.0
    n = 0

    confmat = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    for (
        images,
        masks,
        _
    ) in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(
                AMP and
                DEVICE.type == "cuda"
            )
        ):
            terrain_logits, boundary_logits = (
                model(images)
            )

            loss = criterion(
                terrain_logits,
                boundary_logits,
                masks
            )

            if use_tta:
                terrain_logits = (
                    predict_with_tta(
                        model,
                        images
                    )
                )

        pred = torch.argmax(
            terrain_logits,
            dim=1
        )

        update_confusion_matrix(
            confmat,
            pred,
            masks,
            NUM_CLASSES
        )

        bs = images.size(0)

        running_loss += (
            loss.item() * bs
        )

        n += bs

    metrics = (
        metrics_from_confmat(
            confmat
        )
    )

    metrics["loss"] = (
        running_loss /
        max(n, 1)
    )

    metrics["confmat"] = confmat

    return metrics


# ============================================================
# 23. TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):
    model.train()

    running_loss = 0.0
    n = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, (
        images,
        masks,
        _
    ) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(
                AMP and
                DEVICE.type == "cuda"
            )
        ):
            terrain_logits, boundary_logits = (
                model(images)
            )

            loss = criterion(
                terrain_logits,
                boundary_logits,
                masks
            )

            loss_for_backward = (
                loss /
                GRAD_ACCUM_STEPS
            )

        scaler.scale(
            loss_for_backward
        ).backward()

        if (
            (step + 1) %
            GRAD_ACCUM_STEPS == 0
            or
            (step + 1) ==
            len(loader)
        ):
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        bs = images.size(0)

        running_loss += (
            loss.item() * bs
        )

        n += bs

    return (
        running_loss /
        max(n, 1)
    )


# ============================================================
# 24. OPTIMIZER / SCHEDULER
# ============================================================

encoder_params = list(
    model.encoder.parameters()
)

encoder_ids = {
    id(p)
    for p in encoder_params
}

decoder_params = [
    p
    for p in model.parameters()
    if id(p)
    not in encoder_ids
]

optimizer = torch.optim.AdamW(
    [
        {
            "params": encoder_params,
            "lr": ENCODER_LR
        },
        {
            "params": decoder_params,
            "lr": LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.10,
    end_factor=1.0,
    total_iters=WARMUP_EPOCHS
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, EPOCHS - WARMUP_EPOCHS),
    eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS]
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        AMP and
        DEVICE.type == "cuda"
    )
)


# ============================================================
# 25. PRE-TRAINING SMOKE TEST
# ============================================================

print(
    "\n================ PRE-TRAINING CHECK ================"
)

try:
    test_images, test_masks, _ = (
        next(iter(train_loader))
    )

    test_images = test_images.to(
        DEVICE
    )

    test_masks = test_masks.to(
        DEVICE
    )

    model.train()

    with torch.amp.autocast(
        device_type=DEVICE.type,
        enabled=(
            AMP and
            DEVICE.type == "cuda"
        )
    ):
        test_terrain, test_boundary = (
            model(test_images)
        )

        test_loss = criterion(
            test_terrain,
            test_boundary,
            test_masks
        )

    if not torch.isfinite(
        test_loss
    ):
        raise RuntimeError(
            "Pre-training loss is not finite."
        )

    if (
        test_terrain.shape[1] !=
        NUM_CLASSES
    ):
        raise RuntimeError(
            f"Model returned "
            f"{test_terrain.shape[1]} "
            f"classes; expected "
            f"{NUM_CLASSES}."
        )

    print("Smoke test PASSED")
    print(
        "Input shape :",
        tuple(test_images.shape)
    )
    print(
        "Output shape:",
        tuple(test_terrain.shape)
    )
    print(
        "Initial loss:",
        f"{test_loss.item():.6f}"
    )

except Exception as exc:
    raise RuntimeError(
        "Pre-training check failed. "
        "Training was NOT started.\n"
        f"Original error: {exc}"
    ) from exc


# ============================================================
# 26. MAIN TRAINING LOOP
# ============================================================

best_score = -float("inf")
bad_epochs = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "pixel_acc": [],
    "miou": [],
    "miou_fg": [],
    "dice": [],
    "rare_miou": [],
}

print(
    "\n================ START BINARY Y-NET TRAINING ================"
)

for epoch in range(
    1,
    EPOCHS + 1
):
    epoch_start = time.perf_counter()

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_metrics = validate(
        model,
        val_loader,
        criterion,
        use_tta=False
    )

    scheduler.step()

    miou = val_metrics[
        "miou_all"
    ]

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_metrics["loss"]
    )

    history["pixel_acc"].append(
        val_metrics["pixel_acc"]
    )

    history["miou"].append(
        miou
    )

    history["miou_fg"].append(
        val_metrics["miou_fg"]
    )

    history["dice"].append(
        val_metrics["mdice_fg"]
    )

    history["rare_miou"].append(
        val_metrics["rare_miou"]
    )

    epoch_time = (
        time.perf_counter() -
        epoch_start
    )

    print(
        f"\nEpoch "
        f"{epoch:03d}/{EPOCHS}"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.5f}"
    )

    print(
        f"Val Loss   : "
        f"{val_metrics['loss']:.5f}"
    )

    print(
        f"Pixel Acc  : "
        f"{val_metrics['pixel_acc']:.4f}"
    )

    print(
        f"mIoU       : "
        f"{val_metrics['miou_all']:.4f}"
    )

    print(
        f"FG mIoU    : "
        f"{val_metrics['miou_fg']:.4f}"
    )

    print(
        f"FG Dice    : "
        f"{val_metrics['mdice_fg']:.4f}"
    )

    print(
        f"Rare mIoU  : "
        f"{val_metrics['rare_miou']:.4f}"
    )

    print(
        f"Epoch time : "
        f"{epoch_time:.1f} sec"
    )

    if miou > best_score:
        best_score = miou
        bad_epochs = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
                "epoch": epoch,
                "best_miou": best_score,
                "binary_stats":
                    binary_stats,
                "config": {
                    "IMG_SIZE": IMG_SIZE,
                    "NUM_CLASSES": NUM_CLASSES,
                    "BINARY_EFFICIENTNET_INDICES":
                        BINARY_EFFICIENTNET_INDICES,
                    "BINARY_DECODER":
                        BINARY_DECODER,
                    "BINARY_ATTENTION":
                        BINARY_ATTENTION
                }
            },
            str(CHECKPOINT_PATH)
        )

        print(
            f"*** New best checkpoint saved: "
            f"{best_score:.4f} ***"
        )

    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print(
            f"\nEarly stopping after "
            f"{PATIENCE} epochs without "
            f"mIoU improvement."
        )
        break


print(
    "\n================ TRAINING COMPLETE ================"
)

print(
    "Best validation mIoU:",
    f"{best_score:.4f}"
)

print(
    "Checkpoint:",
    os.path.abspath(str(CHECKPOINT_PATH))
)


# ============================================================
# 27. LOAD BEST CHECKPOINT
# ============================================================

if os.path.exists(
    str(CHECKPOINT_PATH)
):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Best binary Y-Net checkpoint loaded."
    )


# ============================================================
# 28. FINAL EVALUATION
# ============================================================

print(
    "\n================ FINAL EVALUATION ================"
)

final_metrics = validate(
    model,
    val_loader,
    criterion,
    use_tta=USE_TTA
)

final_cm = final_metrics[
    "confmat"
]

print(
    f"Pixel Accuracy : "
    f"{final_metrics['pixel_acc']:.4f}"
)

print(
    f"mIoU (All)     : "
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
    f"Precision (FG): "
    f"{final_metrics['precision'][1:].mean():.4f}"
)

print(
    f"Recall (FG)   : "
    f"{final_metrics['recall'][1:].mean():.4f}"
)

print(
    f"F1 (FG)       : "
    f"{final_metrics['f1'][1:].mean():.4f}"
)

print(
    f"Sensitivity    : "
    f"{final_metrics['recall'][1:].mean():.4f}"
)

print(
    f"Specificity    : "
    f"{final_metrics['specificity_mean']:.4f}"
)

print(
    f"PQ Proxy       : "
    f"{final_metrics['pq_proxy']:.4f}"
)


# ============================================================
# 29. PER-CLASS RESULTS
# ============================================================

print(
    "\n================ PER-CLASS RESULTS ================"
)

for c in range(NUM_CLASSES):
    print(
        f"{c:02d} "
        f"{CLASS_NAMES[c]:10s} | "
        f"IoU={final_metrics['iou'][c]:.4f} | "
        f"Dice={final_metrics['dice'][c]:.4f} | "
        f"Prec={final_metrics['precision'][c]:.4f} | "
        f"Recall={final_metrics['recall'][c]:.4f} | "
        f"F1={final_metrics['f1'][c]:.4f} | "
        f"Spec={final_metrics['specificity'][c]:.4f}"
    )


# ============================================================
# 30. SAVE CONFUSION MATRIX
# ============================================================

np.save(
    os.path.join(
        EXTENDED_RESULTS_DIR,
        "binary_YNet_confusion_matrix.npy"
    ),
    final_cm
)

plt.figure(
    figsize=(10, 9)
)

plt.imshow(
    final_cm,
    cmap="Blues"
)

plt.colorbar()

plt.title(
    "Binary Y-Net Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Ground Truth Class"
)

plt.xticks(
    range(NUM_CLASSES)
)

plt.yticks(
    range(NUM_CLASSES)
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EXTENDED_RESULTS_DIR,
        "binary_YNet_confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 31. SD ESTIMATION
# ============================================================
# Dataset-level metrics above are deterministic from the complete
# validation confusion matrix. To estimate variation across images,
# compute per-image confusion matrices and corresponding metrics.

@torch.no_grad()
def per_image_metrics(
    model,
    loader
):
    model.eval()

    records = []

    for images, masks, paths in loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        if USE_TTA:
            logits = predict_with_tta(
                model,
                images
            )
        else:
            logits, _ = model(
                images
            )

        pred = torch.argmax(
            logits,
            dim=1
        )

        for i in range(
            images.size(0)
        ):
            cm = np.zeros(
                (
                    NUM_CLASSES,
                    NUM_CLASSES
                ),
                dtype=np.int64
            )

            update_confusion_matrix(
                cm,
                pred[i],
                masks[i],
                NUM_CLASSES
            )

            m = metrics_from_confmat(
                cm
            )

            records.append(
                {
                    "path": paths[i],
                    "pixel_acc":
                        m["pixel_acc"],
                    "miou":
                        m["miou_all"],
                    "miou_fg":
                        m["miou_fg"],
                    "dice_fg":
                        m["mdice_fg"],
                    "precision_fg":
                        float(
                            m["precision"][1:].mean()
                        ),
                    "recall_fg":
                        float(
                            m["recall"][1:].mean()
                        ),
                    "f1_fg":
                        float(
                            m["f1"][1:].mean()
                        ),
                    "sensitivity":
                        float(
                            m["recall"][1:].mean()
                        ),
                    "specificity":
                        m["specificity_mean"],
                    "pq":
                        m["pq_proxy"]
                }
            )

    return records


print(
    "\n================ COMPUTING PER-IMAGE SD ================"
)

image_metric_records = per_image_metrics(
    model,
    val_loader
)

metric_keys = [
    "pixel_acc",
    "miou",
    "miou_fg",
    "dice_fg",
    "precision_fg",
    "recall_fg",
    "f1_fg",
    "sensitivity",
    "specificity",
    "pq"
]

sd_results = {}

for key in metric_keys:
    values = np.array(
        [
            r[key]
            for r in image_metric_records
        ],
        dtype=np.float64
    )

    sd_results[key] = (
        float(np.std(
            values,
            ddof=1
        ))
        if len(values) > 1
        else 0.0
    )

print(
    f"Pixel Accuracy : "
    f"{final_metrics['pixel_acc']:.4f} "
    f"± {sd_results['pixel_acc']:.4f}"
)

print(
    f"mIoU           : "
    f"{final_metrics['miou_all']:.4f} "
    f"± {sd_results['miou']:.4f}"
)

print(
    f"FG mIoU        : "
    f"{final_metrics['miou_fg']:.4f} "
    f"± {sd_results['miou_fg']:.4f}"
)

print(
    f"Dice (FG)      : "
    f"{final_metrics['mdice_fg']:.4f} "
    f"± {sd_results['dice_fg']:.4f}"
)

print(
    f"Precision (FG) : "
    f"{final_metrics['precision'][1:].mean():.4f} "
    f"± {sd_results['precision_fg']:.4f}"
)

print(
    f"Recall (FG)    : "
    f"{final_metrics['recall'][1:].mean():.4f} "
    f"± {sd_results['recall_fg']:.4f}"
)

print(
    f"F1 (FG)        : "
    f"{final_metrics['f1'][1:].mean():.4f} "
    f"± {sd_results['f1_fg']:.4f}"
)

print(
    f"Sensitivity     : "
    f"{final_metrics['recall'][1:].mean():.4f} "
    f"± {sd_results['sensitivity']:.4f}"
)

print(
    f"Specificity     : "
    f"{final_metrics['specificity_mean']:.4f} "
    f"± {sd_results['specificity']:.4f}"
)

print(
    f"PQ Proxy        : "
    f"{final_metrics['pq_proxy']:.4f} "
    f"± {sd_results['pq']:.4f}"
)


# ============================================================
# 32. COMPUTATIONAL PERFORMANCE
# ============================================================

def denormalize(images):
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=images.device
    ).view(
        1, 3, 1, 1
    )

    std = torch.tensor(
        IMAGENET_STD,
        device=images.device
    ).view(
        1, 3, 1, 1
    )

    return (
        images * std + mean
    ).clamp(
        0,
        1
    )


def benchmark_inference(
    model,
    loader,
    num_images=50
):
    model.eval()

    times = []

    processed = 0

    process = psutil.Process(
        os.getpid()
    )

    cpu_samples = []
    ram_samples = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )

    with torch.no_grad():
        for images, _, _ in loader:
            if processed >= num_images:
                break

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            for _ in range(3):
                _ = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() -
                start
            )

            bs = images.size(0)

            per_image_ms = (
                elapsed /
                bs *
                1000.0
            )

            times.append(
                per_image_ms
            )

            cpu_samples.append(
                psutil.cpu_percent(
                    interval=None
                )
            )

            ram_samples.append(
                process.memory_info()
                .rss /
                1024 /
                1024
            )

            processed += bs

    if not times:
        return {}

    times = np.array(
        times,
        dtype=np.float64
    )

    cpu_samples = np.array(
        cpu_samples,
        dtype=np.float64
    )

    ram_samples = np.array(
        ram_samples,
        dtype=np.float64
    )

    mean_ms = float(
        times.mean()
    )

    sd_ms = float(
        times.std(
            ddof=1
        )
    ) if len(times) > 1 else 0.0

    fps_values = (
        1000.0 /
        np.maximum(
            times,
            1e-9
        )
    )

    gpu_memory_mb = None

    if DEVICE.type == "cuda":
        gpu_memory_mb = (
            torch.cuda.max_memory_allocated(
                DEVICE
            ) /
            1024 /
            1024
        )

    return {
        "inference_ms":
            mean_ms,
        "inference_sd":
            sd_ms,
        "fps":
            float(
                fps_values.mean()
            ),
        "fps_sd":
            float(
                fps_values.std(
                    ddof=1
                )
            )
            if len(fps_values) > 1
            else 0.0,
        "cpu":
            float(
                cpu_samples.mean()
            ),
        "cpu_sd":
            float(
                cpu_samples.std(
                    ddof=1
                )
            )
            if len(cpu_samples) > 1
            else 0.0,
        "ram_mb":
            float(
                ram_samples.mean()
            ),
        "ram_sd":
            float(
                ram_samples.std(
                    ddof=1
                )
            )
            if len(ram_samples) > 1
            else 0.0,
        "gpu_memory_mb":
            gpu_memory_mb
    }


print(
    "\n================ COMPUTATIONAL PERFORMANCE ================"
)

perf = benchmark_inference(
    model,
    val_loader,
    num_images=50
)

print(
    f"Inference Time : "
    f"{perf['inference_ms']:.4f} "
    f"± {perf['inference_sd']:.4f} ms/image"
)

print(
    f"FPS            : "
    f"{perf['fps']:.2f} "
    f"± {perf['fps_sd']:.2f}"
)

print(
    f"CPU Usage      : "
    f"{perf['cpu']:.2f} "
    f"± {perf['cpu_sd']:.2f}%"
)

print(
    f"RAM Usage      : "
    f"{perf['ram_mb']:.2f} "
    f"± {perf['ram_sd']:.2f} MB"
)

if perf["gpu_memory_mb"] is not None:
    print(
        f"Peak GPU Memory: "
        f"{perf['gpu_memory_mb']:.2f} MB"
    )


# ============================================================
# 33. FLOPs
# ============================================================

def estimate_flops(model):
    try:
        from torch.profiler import (
            profile,
            ProfilerActivity
        )

        dummy = torch.randn(
            1,
            3,
            IMG_SIZE,
            IMG_SIZE,
            device=DEVICE
        )

        model.eval()

        with torch.no_grad():
            with profile(
                activities=[
                    ProfilerActivity.CPU
                ],
                record_shapes=True,
                with_flops=True
            ) as prof:
                model(dummy)

        total_flops = 0.0

        for evt in prof.key_averages():
            if hasattr(
                evt,
                "flops"
            ):
                total_flops += (
                    evt.flops or 0
                )

        if total_flops > 0:
            return (
                total_flops /
                1e9
            )

    except Exception as exc:
        print(
            "FLOPs profiling unavailable:",
            exc
        )

    return None


gflops = estimate_flops(
    model
)

if gflops is not None:
    print(
        f"Estimated FP-operation "
        f"equivalent: "
        f"{gflops:.4f} GFLOPs"
    )
else:
    print(
        "Estimated FLOPs: unavailable"
    )


# ============================================================
# 34. VISUALIZATION HELPERS
# ============================================================

def labels_to_rgb(
    labels
):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    rgb = np.zeros(
        (
            labels.shape[0],
            labels.shape[1],
            3
        ),
        dtype=np.uint8
    )

    for c, color in enumerate(
        PALETTE_RGB
    ):
        rgb[
            labels == c
        ] = color

    return rgb


def save_prediction_visualizations(
    model,
    loader,
    max_images=30
):
    os.makedirs(
        EXTENDED_RESULTS_DIR,
        exist_ok=True
    )

    model.eval()

    saved = 0

    for images, masks, paths in loader:
        if saved >= max_images:
            break

        images_device = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.no_grad():
            if USE_TTA:
                logits = (
                    predict_with_tta(
                        model,
                        images_device
                    )
                )
            else:
                logits, _ = model(
                    images_device
                )

            predictions = torch.argmax(
                logits,
                dim=1
            )

        originals = (
            denormalize(
                images_device
            )
            .cpu()
            .numpy()
        )

        gt = masks.cpu().numpy()
        pred = (
            predictions
            .cpu()
            .numpy()
        )

        for i in range(
            len(paths)
        ):
            if saved >= max_images:
                break

            raw = np.transpose(
                originals[i],
                (1, 2, 0)
            )

            raw = np.clip(
                raw,
                0,
                1
            )

            gt_rgb = labels_to_rgb(
                gt[i]
            )

            pred_rgb = labels_to_rgb(
                pred[i]
            )

            overlay = (
                0.55 * raw +
                0.45 * (
                    pred_rgb / 255.0
                )
            )

            base = Path(
                paths[i]
            ).stem

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(20, 5)
            )

            axes[0].imshow(raw)
            axes[0].set_title(
                "Original"
            )
            axes[0].axis("off")

            axes[1].imshow(gt_rgb)
            axes[1].set_title(
                "Ground Truth"
            )
            axes[1].axis("off")

            axes[2].imshow(pred_rgb)
            axes[2].set_title(
                "Binary Y-Net Prediction"
            )
            axes[2].axis("off")

            axes[3].imshow(overlay)
            axes[3].set_title(
                "Prediction Overlay"
            )
            axes[3].axis("off")

            fig.suptitle(
                f"Binary Y-Net | {base}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    EXTENDED_RESULTS_DIR,
                    f"{saved + 1:02d}_{base}_"
                    f"binary_YNet_result.png"
                ),
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            saved += 1

    return saved


print(
    "\n================ SAVING VALIDATION VISUALIZATIONS ================"
)

visualizations_saved = (
    save_prediction_visualizations(
        model,
        val_loader,
        MAX_VISUALIZATIONS
    )
)

print(
    "Visualizations saved:",
    visualizations_saved
)


# ============================================================
# 35. TRAINING CURVES
# ============================================================

def save_training_curves(
    history
):
    if not history["train_loss"]:
        return

    epochs = np.arange(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Binary Y-Net Training/Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EXTENDED_RESULTS_DIR,
            "binary_YNet_loss_curve.png"
        ),
        dpi=180
    )

    plt.close()

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["miou"],
        label="mIoU"
    )

    plt.plot(
        epochs,
        history["miou_fg"],
        label="FG mIoU"
    )

    plt.plot(
        epochs,
        history["dice"],
        label="FG Dice"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Binary Y-Net Validation Metrics"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EXTENDED_RESULTS_DIR,
            "binary_YNet_metric_curve.png"
        ),
        dpi=180
    )

    plt.close()


save_training_curves(
    history
)


# ============================================================
# 36. SEGMENTATION GRAD-CAM
# ============================================================

class SegmentationGradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):
        self.model = model
        self.target_layer = (
            target_layer
        )

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

        predicted_classes = (
            probabilities.argmax(
                dim=1
            )
        )

        predicted_confidence = (
            probabilities.gather(
                1,
                predicted_classes.unsqueeze(
                    1
                )
            )
            .squeeze(1)
        )

        score = (
            predicted_confidence.mean()
        )

        score.backward(
            retain_graph=True
        )

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights *
            activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam_min = (
            cam.flatten(1)
            .min(dim=1)[0]
            .view(
                -1,
                1,
                1,
                1
            )
        )

        cam_max = (
            cam.flatten(1)
            .max(dim=1)[0]
            .view(
                -1,
                1,
                1,
                1
            )
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max -
            cam_min +
            1e-8
        )

        return (
            cam.detach(),
            predicted_classes.detach(),
            predicted_confidence.detach()
        )


print(
    "\n================ GENERATING HEATMAPS ================"
)

try:
    target_layer = (
        model.encoder.features[-1]
    )

    grad_cam = (
        SegmentationGradCAM(
            model,
            target_layer
        )
    )

    model.eval()

    heatmaps_saved = 0

    for images, masks, paths in val_loader:

        if (
            heatmaps_saved >=
            MAX_HEATMAP_IMAGES
        ):
            break

        images_device = images.to(
            DEVICE,
            non_blocking=True
        )

        cam, predictions, confidence = (
            grad_cam.generate(
                images_device
            )
        )

        original = (
            denormalize(
                images_device
            )
            .detach()
            .cpu()
        )

        cam = cam.cpu()
        predictions = predictions.cpu()
        confidence = confidence.cpu()

        for i in range(
            images.size(0)
        ):
            if (
                heatmaps_saved >=
                MAX_HEATMAP_IMAGES
            ):
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

            prediction = (
                predictions[i]
                .numpy()
            )

            base_name = Path(
                paths[i]
            ).stem

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
                "Binary Y-Net Segmentation Heatmap"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    HEATMAP_DIR,
                    f"{heatmaps_saved + 1:02d}_"
                    f"{base_name}_heatmap.png"
                ),
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            plt.figure(
                figsize=(7, 6)
            )

            plt.imshow(raw)

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
                "Binary Y-Net Heatmap Overlay"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    HEATMAP_DIR,
                    f"{heatmaps_saved + 1:02d}_"
                    f"{base_name}_overlay.png"
                ),
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            pred_rgb = labels_to_rgb(
                prediction
            )

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 6)
            )

            axes[0].imshow(raw)
            axes[0].set_title(
                "Original Image"
            )
            axes[0].axis("off")

            axes[1].imshow(raw)
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

            axes[2].imshow(pred_rgb)
            axes[2].set_title(
                "Predicted Segmentation"
            )
            axes[2].axis("off")

            fig.suptitle(
                "Binary Y-Net | "
                f"Mean Confidence: "
                f"{confidence[i].mean().item():.4f}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    HEATMAP_DIR,
                    f"{heatmaps_saved + 1:02d}_"
                    f"{base_name}_complete.png"
                ),
                dpi=180,
                bbox_inches="tight"
            )

            plt.close()

            heatmaps_saved += 1

    grad_cam.remove_hooks()

    print(
        "Heatmaps generated:",
        heatmaps_saved
    )

except Exception as exc:
    print(
        "WARNING: Heatmap generation skipped:",
        exc
    )


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("BINARY Y-NET FINAL SUMMARY")
print("=" * 80)

print(
    f"Best validation mIoU     : "
    f"{best_score:.4f}"
)

print(
    f"Final pixel accuracy     : "
    f"{final_metrics['pixel_acc']:.4f} "
    f"± {sd_results['pixel_acc']:.4f}"
)

print(
    f"Final mIoU                : "
    f"{final_metrics['miou_all']:.4f} "
    f"± {sd_results['miou']:.4f}"
)

print(
    f"Final FG mIoU             : "
    f"{final_metrics['miou_fg']:.4f} "
    f"± {sd_results['miou_fg']:.4f}"
)

print(
    f"Final FG Dice             : "
    f"{final_metrics['mdice_fg']:.4f} "
    f"± {sd_results['dice_fg']:.4f}"
)

print(
    f"Final FG Precision        : "
    f"{final_metrics['precision'][1:].mean():.4f} "
    f"± {sd_results['precision_fg']:.4f}"
)

print(
    f"Final FG Recall           : "
    f"{final_metrics['recall'][1:].mean():.4f} "
    f"± {sd_results['recall_fg']:.4f}"
)

print(
    f"Final FG F1               : "
    f"{final_metrics['f1'][1:].mean():.4f} "
    f"± {sd_results['f1_fg']:.4f}"
)

print(
    f"Sensitivity               : "
    f"{final_metrics['recall'][1:].mean():.4f} "
    f"± {sd_results['sensitivity']:.4f}"
)

print(
    f"Specificity               : "
    f"{final_metrics['specificity_mean']:.4f} "
    f"± {sd_results['specificity']:.4f}"
)

print(
    f"PQ Proxy                  : "
    f"{final_metrics['pq_proxy']:.4f} "
    f"± {sd_results['pq']:.4f}"
)

print()
print(
    f"Total parameters          : "
    f"{binary_stats['total_params']:,}"
)

print(
    f"Binary parameters         : "
    f"{binary_stats['binary_params']:,}"
)

print(
    f"Binary ratio              : "
    f"{binary_stats['binary_ratio'] * 100:.2f}%"
)

print(
    f"Theoretical FP32 size     : "
    f"{binary_stats['fp32_size_mb']:.2f} MB"
)

print(
    f"Theoretical 1-bit size    : "
    f"{binary_stats['theoretical_binary_size_mb']:.2f} MB"
)

print(
    f"Theoretical compression   : "
    f"{binary_stats['theoretical_compression']:.2f}x"
)

if gflops is not None:
    print(
        f"FLOPs equivalent         : "
        f"{gflops:.4f} GFLOPs"
    )

print(
    f"Inference time            : "
    f"{perf['inference_ms']:.4f} "
    f"± {perf['inference_sd']:.4f} ms/image"
)

print(
    f"FPS                       : "
    f"{perf['fps']:.2f} "
    f"± {perf['fps_sd']:.2f}"
)

print(
    f"CPU usage                 : "
    f"{perf['cpu']:.2f} "
    f"± {perf['cpu_sd']:.2f}%"
)

print(
    f"RAM usage                 : "
    f"{perf['ram_mb']:.2f} "
    f"± {perf['ram_sd']:.2f} MB"
)

if perf["gpu_memory_mb"] is not None:
    print(
        f"Peak GPU memory          : "
        f"{perf['gpu_memory_mb']:.2f} MB"
    )

print()
print(
    "Checkpoint:",
    os.path.abspath(str(CHECKPOINT_PATH))
)

print(
    "Validation visualizations:",
    os.path.abspath(str(EXTENDED_RESULTS_DIR))
)

print(
    "Heatmaps:",
    os.path.abspath(str(HEATMAP_DIR))
)

print("=" * 80)

print(
    "\nIMPORTANT: The 0.85 target is not hard-coded "
    "into the metrics. The model must be trained and "
    "measured to determine whether it reaches >=0.85 mIoU."
)

print(
    "IMPORTANT: BinaryConv2d keeps FP32 shadow weights "
    "during training. The reported 1-bit size is theoretical "
    "binary storage; actual runtime speedup requires a backend "
    "with optimized binary/XNOR kernels."
)

print("=" * 80)
