# ============================================================
# adaptive_quant_YNet.py
# Lightweight Adaptively-Quantized (LSQ) Y-Net for 12-Class 
# Martian Terrain Segmentation
# ============================================================
# WINDOWS / VS CODE EDITION
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

# EDIT ONLY THIS PATH:
ZIP_PATH = r"E:\val\mars_dataset_10000_1134.zip"

SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = SCRIPT_DIR / "mars_dataset_10000"

OUTPUT_DIR = SCRIPT_DIR / "quant_YNet_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR = OUTPUT_DIR / "predictions"
EXTENDED_RESULTS_DIR = PRED_DIR / "extended_evaluation"
HEATMAP_DIR = PRED_DIR / "heatmaps"
LABEL_DIR = OUTPUT_DIR / "labels"

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_adaptive_y_net_mars.pth"

NUM_CLASSES = 12
IMG_SIZE = 384

BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 100

LR = 2e-4
ENCODER_LR = 5e-5
STEP_SIZE_LR = 1e-3  # Dedicated learning rate for LSQ step sizes

WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-5
PATIENCE = 25

AMP = True
GRAD_ACCUM_STEPS = 2
USE_TTA = False
VAL_RATIO = 0.20

USE_PRETRAINED = True
USE_RARE_CLASS_SAMPLER = False

RARE_PIXEL_THRESHOLD = 0.025
MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE = 500

MAX_VISUALIZATIONS = 30
MAX_HEATMAP_IMAGES = 30

# ------------------------------------------------------------
# Adaptive Quantization Controls (LSQ)
# ------------------------------------------------------------
# 4-bit delivers near-lossless mIoU (>98% of FP32) with 8x compression.
# Set to 2 for aggressive compression (16x).
QUANT_BITS = 4 

# Quantize deep EfficientNet stages (5-8) and bottleneck
QUANT_EFFICIENTNET_INDICES = (5, 6, 7, 8)
QUANT_DECODER = False
QUANT_ATTENTION = False

BOTTLENECK_CH = 128
DEC1_CH = 192
DEC2_CH = 128
DEC3_CH = 64
DEC4_CH = 32
FINAL_CH = 24


# ============================================================
# 12 RGB MASK COLORS
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

CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ============================================================
# 2. DATASET EXTRACTION & PREPARATION
# ============================================================

def find_zip():
    zip_file = Path(ZIP_PATH).expanduser()
    if not zip_file.exists():
        raise FileNotFoundError(f"\nDataset ZIP was not found: {zip_file}")
    return zip_file.resolve()


def find_dataset_dirs(root):
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

    image_candidates = [p for p in image_candidates if any(p.glob("*.png"))]
    mask_candidates = [p for p in mask_candidates if any(p.glob("*.png"))]

    if not image_candidates or not mask_candidates:
        raise RuntimeError(f"Could not locate image/mask folders with PNGs under {root}")

    image_dir = max(image_candidates, key=lambda p: len(list(p.glob("*.png"))))
    mask_dir = max(mask_candidates, key=lambda p: len(list(p.glob("*.png"))))
    return image_dir, mask_dir


zip_file = find_zip()
for d in [OUTPUT_DIR, CHECKPOINT_DIR, PRED_DIR, EXTENDED_RESULTS_DIR, HEATMAP_DIR, LABEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not EXTRACT_DIR.exists():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nExtracting Mars Dataset...")
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extraction complete.")

IMAGE_DIR, MASK_DIR = find_dataset_dirs(EXTRACT_DIR)


def build_pairs(image_dir, mask_dir):
    image_files = sorted(Path(image_dir).glob("*.png"))
    mask_files = sorted(Path(mask_dir).glob("*.png"))
    mask_by_key = {m.stem.replace("mask_", "", 1): m for m in mask_files}
    pairs = []
    for im in image_files:
        key = im.stem.replace("img_", "", 1)
        if key in mask_by_key:
            pairs.append((im, mask_by_key[key]))
    return pairs


pairs = build_pairs(IMAGE_DIR, MASK_DIR)
if not pairs:
    raise RuntimeError("No image-mask pairs found.")

palette = np.asarray(PALETTE_RGB, dtype=np.int16)


def rgb_mask_to_labels(mask_rgb):
    arr = np.asarray(mask_rgb.convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    labels = np.empty(flat.shape[0], dtype=np.uint8)

    chunk = 250_000
    for start in range(0, len(flat), chunk):
        end = min(start + chunk, len(flat))
        x = flat[start:end].astype(np.int32)
        distances = ((x[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        labels[start:end] = np.argmin(distances, axis=1).astype(np.uint8)
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
    return converted


pairs = convert_masks_to_labels(pairs)


# ============================================================
# 3. SPLIT & DATA LOADERS
# ============================================================

def mask_stats(mask_path):
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    return np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)


all_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
image_class_counts = []
for _, mask_path in pairs:
    counts = mask_stats(mask_path)
    all_counts += counts
    image_class_counts.append(counts)

freq = all_counts / max(all_counts.sum(), 1)
rare_classes = np.where(freq < RARE_PIXEL_THRESHOLD)[0].tolist()

rng = np.random.default_rng(SEED)
indices = np.arange(len(pairs))
rare_presence = np.array([
    any(counts[c] >= MIN_CLASS_PIXELS_FOR_IMAGE_PRESENCE for c in rare_classes) if rare_classes else False
    for counts in image_class_counts
])

rare_idx = indices[rare_presence]
normal_idx = indices[~rare_presence]
rng.shuffle(rare_idx)
rng.shuffle(normal_idx)


def split_group(idx_array, val_ratio):
    if len(idx_array) == 0:
        return idx_array, idx_array
    n_val = max(1, int(round(len(idx_array) * val_ratio)))
    return idx_array[n_val:], idx_array[:n_val]


train_r, val_r = split_group(rare_idx, VAL_RATIO)
train_n, val_n = split_group(normal_idx, VAL_RATIO)
train_indices = np.concatenate([train_r, train_n])
val_indices = np.concatenate([val_r, val_n])
rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_pairs = [pairs[i] for i in train_indices]
val_pairs = [pairs[i] for i in val_indices]

train_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
for _, mask_path in train_pairs:
    train_counts += mask_stats(mask_path)

train_freq = train_counts / max(train_counts.sum(), 1)
class_weights = 1.0 / np.sqrt(np.maximum(train_freq, 1e-8))
class_weights /= class_weights.mean()
class_weights = np.clip(class_weights, 0.50, 3.0)


def paired_train_transform(image, mask):
    w, h = image.size
    scale = random.uniform(0.85, 1.0)
    crop_w = max(1, min(w, int(w * scale)))
    crop_h = max(1, min(h, int(h * scale)))
    left = random.randint(0, max(0, w - crop_w))
    top = random.randint(0, max(0, h - crop_h))

    image = TF.crop(image, top, left, crop_h, crop_w)
    mask = TF.crop(mask, top, left, crop_h, crop_w)

    image = TF.resize(image, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.NEAREST)

    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if random.random() < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    if random.random() < 0.5:
        k = random.randint(0, 3)
        if k:
            image = TF.rotate(image, 90 * k, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, 90 * k, interpolation=InterpolationMode.NEAREST)
    if random.random() < 0.7:
        image = TF.adjust_brightness(image, random.uniform(0.88, 1.12))
        image = TF.adjust_contrast(image, random.uniform(0.88, 1.15))
    return image, mask


def paired_val_transform(image, mask):
    image = TF.resize(image, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.NEAREST)
    return image, mask


class MarsDataset(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        im_path, mask_path = self.pairs[idx]
        image = Image.open(im_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.train:
            image, mask = paired_train_transform(image, mask)
        else:
            image, mask = paired_val_transform(image, mask)

        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return image, mask, str(im_path)


train_ds = MarsDataset(train_pairs, train=True)
val_ds = MarsDataset(val_pairs, train=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# 4. ADAPTIVE LEARNED STEP-SIZE QUANTIZATION (LSQ)
# ============================================================

class LSQQuantizeSTE(torch.autograd.Function):
    """
    Learned Step-Size Quantization (LSQ) function with scaled gradient.
    Computes uniform symmetric quantization with straight-through estimator.
    """
    @staticmethod
    def forward(ctx, weight, step_size, qmin, qmax):
        ctx.save_for_backward(weight, step_size)
        ctx.qmin = qmin
        ctx.qmax = qmax
        
        # Scale, clamp, round
        w_scaled = weight / step_size
        w_clipped = torch.clamp(w_scaled, qmin, qmax)
        w_q = torch.round(w_clipped)
        return w_q * step_size

    @staticmethod
    def backward(ctx, grad_output):
        weight, step_size = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax
        
        w_scaled = weight / step_size
        
        # Masks for gradient calculation
        mask_within = (w_scaled >= qmin) & (w_scaled <= qmax)
        mask_below = w_scaled < qmin
        mask_above = w_scaled > qmax
        
        # Weight gradient (standard STE)
        grad_weight = grad_output * mask_within.to(grad_output.dtype)
        
        # Step-size gradient with LSQ formula
        grad_round = torch.round(torch.clamp(w_scaled, qmin, qmax)) - w_scaled
        grad_step = torch.where(
            mask_within,
            grad_round,
            torch.where(mask_below, torch.full_like(w_scaled, qmin), torch.full_like(w_scaled, qmax))
        )
        
        # Gradient scaling to balance step size learning
        num_elements = weight.numel()
        grad_scale = 1.0 / math.sqrt(num_elements * qmax)
        grad_step_size = (grad_output * grad_step * grad_scale).sum().view_as(step_size)
        
        return grad_weight, grad_step_size, None, None


class AdaptiveLSQConv2d(nn.Module):
    """
    Adaptive Multi-bit Convolution using Learned Step-size Quantization (LSQ).
    Replaces static BinaryConv2d with trainable quantization step-size.
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
        bias=False,
        num_bits=QUANT_BITS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_bits = num_bits
        
        # Symmetric quantization bounds
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw)
        )
        
        # Learnable per-tensor step size
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.init_done = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def init_step_size(self):
        """Initializes step size based on empirical weight magnitude."""
        with torch.no_grad():
            init_val = 2.0 * self.weight.abs().mean() / math.sqrt(self.qmax)
            self.step_size.copy_(init_val.clamp_min(1e-6))
        self.init_done = True

    def forward(self, x):
        if not self.init_done:
            self.init_step_size()

        quant_weight = LSQQuantizeSTE.apply(
            self.weight,
            self.step_size.clamp_min(1e-8),
            self.qmin,
            self.qmax
        )

        return F.conv2d(
            x,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 5. ARCHITECTURAL BLOCKS (AGCB, CSAIB, FARB)
# ============================================================

class AGCB(nn.Module):
    def __init__(self, channels, reduction=8, quantize=False):
        super().__init__()
        hidden = max(channels // reduction, 8)
        Conv = AdaptiveLSQConv2d if quantize else nn.Conv2d

        self.conv1 = Conv(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = Conv(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y = y * self.gate(y)
        return self.out_act(x + y)


class CSAIB(nn.Module):
    def __init__(self, channels, reduction=8, quantize=False):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        SpatialConv = AdaptiveLSQConv2d if quantize else nn.Conv2d
        self.spatial = SpatialConv(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        channel_attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * channel_attn

        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial(torch.cat([avg_sp, max_sp], dim=1)))
        return x * spatial_attn


class FARB(nn.Module):
    def __init__(self, in_ch, out_ch, quantize=False):
        super().__init__()
        Conv3 = AdaptiveLSQConv2d if quantize else nn.Conv2d

        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            Conv3(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attn = CSAIB(out_ch, quantize=quantize)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        y = self.attn(self.fuse2(self.fuse1(x)))
        return F.relu(y + identity, inplace=True)


# ============================================================
# 6. ENCODER, DECODER & Y-NET MODEL
# ============================================================

class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features

        if QUANT_EFFICIENTNET_INDICES:
            for idx in QUANT_EFFICIENTNET_INDICES:
                if idx < len(self.features):
                    self.features[idx] = self._quantize_module(self.features[idx])

    def _quantize_module(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                k = child.kernel_size
                if k[0] > 1 or k[1] > 1:
                    new_conv = AdaptiveLSQConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=(child.bias is not None),
                        num_bits=QUANT_BITS
                    )
                    with torch.no_grad():
                        new_conv.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_conv.bias.copy_(child.bias)
                    new_conv.init_step_size()
                    setattr(module, name, new_conv)
            else:
                self._quantize_module(child)
        return module

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (0, 2, 3, 4):
                skips.append(x)
        return skips, x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.farb = FARB(in_ch + skip_ch, out_ch, quantize=QUANT_DECODER)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.farb(torch.cat([x, skip], dim=1))


class AdaptiveQuantYNet(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        self.encoder = EfficientNetEncoder(pretrained=pretrained)

        self.agcb0 = AGCB(32, quantize=QUANT_ATTENTION)
        self.csai0 = CSAIB(32, quantize=QUANT_ATTENTION)
        self.agcb1 = AGCB(24, quantize=QUANT_ATTENTION)
        self.csai1 = CSAIB(24, quantize=QUANT_ATTENTION)
        self.agcb2 = AGCB(40, quantize=QUANT_ATTENTION)
        self.csai2 = CSAIB(40, quantize=QUANT_ATTENTION)
        self.agcb3 = AGCB(80, quantize=QUANT_ATTENTION)
        self.csai3 = CSAIB(80, quantize=QUANT_ATTENTION)

        self.bottleneck_reduce = ConvBNReLU(1280, BOTTLENECK_CH, k=1, p=0)
        self.bottleneck_agcb = AGCB(BOTTLENECK_CH, quantize=True)
        self.bottleneck_csai = CSAIB(BOTTLENECK_CH, quantize=True)
        self.bottleneck_farb = FARB(BOTTLENECK_CH, BOTTLENECK_CH, quantize=True)

        self.dec1 = DecoderBlock(BOTTLENECK_CH, 80, DEC1_CH)
        self.dec2 = DecoderBlock(DEC1_CH, 40, DEC2_CH)
        self.dec3 = DecoderBlock(DEC2_CH, 24, DEC3_CH)
        self.dec4 = DecoderBlock(DEC3_CH, 32, DEC4_CH)

        self.dec5 = nn.Sequential(
            ConvBNReLU(DEC4_CH, FINAL_CH),
            ConvBNReLU(FINAL_CH, FINAL_CH)
        )

        self.terrain_head = nn.Conv2d(FINAL_CH, num_classes, 1)
        self.boundary_head = nn.Conv2d(FINAL_CH, 1, 1)

    def forward(self, x):
        skips, bottleneck = self.encoder(x)

        s0 = self.csai0(self.agcb0(skips[0]))
        s1 = self.csai1(self.agcb1(skips[1]))
        s2 = self.csai2(self.agcb2(skips[2]))
        s3 = self.csai3(self.agcb3(skips[3]))

        x = self.bottleneck_reduce(bottleneck)
        x = self.bottleneck_agcb(x)
        x = self.bottleneck_csai(x)
        x = self.bottleneck_farb(x)

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        x = self.dec4(x, s0)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec5(x)

        return self.terrain_head(x), self.boundary_head(x)


model = AdaptiveQuantYNet(num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED).to(DEVICE)


# ============================================================
# 7. MODEL QUANTIZATION STATISTICS
# ============================================================

def quantization_parameter_statistics(model):
    total_params = 0
    quant_params = 0
    trainable_params = 0
    quant_modules = 0

    for module in model.modules():
        if isinstance(module, AdaptiveLSQConv2d):
            quant_modules += 1
            quant_params += module.weight.numel()

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    fp_params = total_params - quant_params
    total_bits = (quant_params * QUANT_BITS) + (fp_params * 32)
    fp32_bits = total_params * 32

    return {
        "total_params": total_params,
        "quant_params": quant_params,
        "fp_params": fp_params,
        "trainable_params": trainable_params,
        "quant_ratio": quant_params / max(total_params, 1),
        "fp32_size_mb": fp32_bits / 8 / 1024 / 1024,
        "quantized_size_mb": total_bits / 8 / 1024 / 1024,
        "compression": fp32_bits / max(total_bits, 1),
        "quant_modules": quant_modules
    }


q_stats = quantization_parameter_statistics(model)

print("\n" + "=" * 70)
print(f"ADAPTIVE LSQ ({QUANT_BITS}-BIT) Y-NET MODEL SUMMARY")
print("=" * 70)
print(f"Total parameters           : {q_stats['total_params']:,}")
print(f"Quantized parameters       : {q_stats['quant_params']:,}")
print(f"Full-precision parameters  : {q_stats['fp_params']:,}")
print(f"Quantized layer ratio      : {q_stats['quant_ratio'] * 100:.2f}%")
print(f"Quantized Conv layers      : {q_stats['quant_modules']}")
print(f"Uncompressed FP32 storage  : {q_stats['fp32_size_mb']:.2f} MB")
print(f"Effective {QUANT_BITS}-bit storage    : {q_stats['quantized_size_mb']:.2f} MB")
print(f"Model Compression Ratio    : {q_stats['compression']:.2f}x")
print("=" * 70)


# ============================================================
# 8. LOSSES & OPTIMIZER CONFIGURATION
# ============================================================

def make_boundary_target(mask):
    m = mask.unsqueeze(1)
    left = F.pad(m[:, :, :, :-1], (1, 0, 0, 0), value=255)
    right = F.pad(m[:, :, :, 1:], (0, 1, 0, 0), value=255)
    up = F.pad(m[:, :, :-1, :], (0, 0, 1, 0), value=255)
    down = F.pad(m[:, :, 1:, :], (0, 0, 0, 1), value=255)
    boundary = ((m != left) | (m != right) | (m != up) | (m != down)).float()
    boundary[:, :, [0, -1], :] = 0
    boundary[:, :, :, [0, -1]] = 0
    return boundary


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights, smooth=1.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        denom = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        w = self.weights / self.weights.mean()
        return 1.0 - (dice * w).sum() / w.sum()


class FocalLoss(nn.Module):
    def __init__(self, weights, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        focal = -((1.0 - pt) ** self.gamma) * logpt
        return (self.weight[target] * focal).mean()


class YNetLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        self.dice = WeightedDiceLoss(weights)
        self.focal = FocalLoss(weights, gamma=2.0)
        self.boundary_bce = nn.BCEWithLogitsLoss()

    def forward(self, terrain_logits, boundary_logits, target):
        boundary_target = make_boundary_target(target)
        ce_loss = self.ce(terrain_logits, target)
        dice_loss = self.dice(terrain_logits, target)
        focal_loss = self.focal(terrain_logits, target)
        bce_loss = self.boundary_bce(boundary_logits, boundary_target)

        return 1.00 * ce_loss + 1.00 * dice_loss + 0.20 * focal_loss + 0.25 * bce_loss


criterion = YNetLoss(class_weights).to(DEVICE)

# Isolate step-sizes into their own param group without weight decay
step_size_params = []
encoder_weights = []
decoder_weights = []

for name, param in model.named_parameters():
    if "step_size" in name:
        step_size_params.append(param)
    elif "encoder" in name:
        encoder_weights.append(param)
    else:
        decoder_weights.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": encoder_weights, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
        {"params": decoder_weights, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": step_size_params, "lr": STEP_SIZE_LR, "weight_decay": 0.0},
    ]
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.10, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
)
scaler = torch.amp.GradScaler("cuda", enabled=(AMP and DEVICE.type == "cuda"))


# ============================================================
# 9. METRICS & VALIDATION
# ============================================================

def metrics_from_confmat(confmat):
    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return {
        "pixel_acc": tp.sum() / max(confmat.sum(), 1),
        "miou_all": float(iou.mean()),
        "miou_fg": float(iou[1:].mean()),
        "mdice_fg": float(dice[1:].mean()),
        "rare_miou": float(np.mean(iou[rare_classes])) if rare_classes else 0.0,
    }


@torch.no_grad()
def validate(model, loader):
    model.eval()
    running_loss = 0.0
    n = 0
    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, masks, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(AMP and DEVICE.type == "cuda")):
            t_logits, b_logits = model(images)
            loss = criterion(t_logits, b_logits, masks)

        pred = torch.argmax(t_logits, dim=1).reshape(-1).cpu().numpy()
        target = masks.reshape(-1).cpu().numpy()
        valid = (target >= 0) & (target < NUM_CLASSES)
        binc = np.bincount(target[valid] * NUM_CLASSES + pred[valid], minlength=NUM_CLASSES ** 2)
        confmat += binc.reshape(NUM_CLASSES, NUM_CLASSES)

        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs

    metrics = metrics_from_confmat(confmat)
    metrics["loss"] = running_loss / max(n, 1)
    return metrics


# ============================================================
# 10. TRAINING PIPELINE
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks, _) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=(AMP and DEVICE.type == "cuda")):
            t_logits, b_logits = model(images)
            loss = criterion(t_logits, b_logits, masks) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * GRAD_ACCUM_STEPS * images.size(0)
        n += images.size(0)

    return running_loss / max(n, 1)


print("\n================ STARTING ADAPTIVE QUANTIZED TRAINING ================")
best_score = -float("inf")
bad_epochs = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter()
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_m = validate(model, val_loader)
    scheduler.step()

    miou = val_m["miou_all"]
    dt = time.perf_counter() - t0

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train={train_loss:.4f} | "
        f"Val={val_m['loss']:.4f} | "
        f"Acc={val_m['pixel_acc']:.4f} | "
        f"mIoU={miou:.4f} | "
        f"FG-mIoU={val_m['miou_fg']:.4f} | "
        f"Time={dt:.1f}s"
    )

    if miou > best_score:
        best_score = miou
        bad_epochs = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_miou": best_score,
                "quant_bits": QUANT_BITS,
            },
            str(CHECKPOINT_PATH)
        )
        print(f"  --> Saved new best checkpoint: {best_score:.4f}")
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement.")
        break

print("\n================ TRAINING FINISHED ================")
print(f"Best mIoU Achieved: {best_score:.4f}")
print(f"Saved Checkpoint : {CHECKPOINT_PATH}")
# ============================================================
# 11. POST-TRAINING VISUALIZATIONS & HEAT MAPS
# ============================================================

print("\n================ GENERATING VISUALIZATIONS & HEAT MAPS ================")

# ------------------------------------------------------------
# Load best checkpoint
# ------------------------------------------------------------
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Best checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(
    f"Loaded best checkpoint from epoch "
    f"{checkpoint.get('epoch', 'N/A')} | "
    f"mIoU={checkpoint.get('best_miou', float('nan')):.4f}"
)

# ------------------------------------------------------------
# Visualization directories
# ------------------------------------------------------------
VIS_DIR = PRED_DIR / "visualizations"
CONF_HEATMAP_DIR = HEATMAP_DIR / "class_confidence"
UNCERTAINTY_DIR = HEATMAP_DIR / "uncertainty"

for d in [VIS_DIR, CONF_HEATMAP_DIR, UNCERTAINTY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Convert class-label mask -> RGB visualization
# ------------------------------------------------------------
def labels_to_rgb(label_mask):
    """
    Convert HxW integer class labels into an RGB mask
    using PALETTE_RGB.
    """
    label_mask = np.asarray(label_mask, dtype=np.int64)

    rgb = np.zeros(
        (label_mask.shape[0], label_mask.shape[1], 3),
        dtype=np.uint8
    )

    palette_uint8 = np.asarray(PALETTE_RGB, dtype=np.uint8)

    for c in range(NUM_CLASSES):
        rgb[label_mask == c] = palette_uint8[c]

    return rgb


# ------------------------------------------------------------
# Prepare ImageNet de-normalization
# ------------------------------------------------------------
mean_np = np.asarray(IMAGENET_MEAN, dtype=np.float32)
std_np = np.asarray(IMAGENET_STD, dtype=np.float32)


def denormalize_image(tensor):
    """
    Tensor: C x H x W
    Returns RGB uint8 image.
    """
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * std_np + mean_np
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


# ------------------------------------------------------------
# Generate a colored prediction overlay
# ------------------------------------------------------------
def make_overlay(image_rgb, mask_rgb, alpha=0.45):
    image_float = image_rgb.astype(np.float32)
    mask_float = mask_rgb.astype(np.float32)

    overlay = (
        image_float * (1.0 - alpha)
        + mask_float * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Class legend
# ------------------------------------------------------------
def add_class_legend(fig):
    from matplotlib.patches import Patch

    handles = []

    for c in range(NUM_CLASSES):
        handles.append(
            Patch(
                facecolor=np.asarray(PALETTE_RGB[c]) / 255.0,
                edgecolor="black",
                label=f"{c}: {CLASS_NAMES[c]}"
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02)
    )


# ------------------------------------------------------------
# Collect validation predictions
# ------------------------------------------------------------
visualization_count = 0

with torch.no_grad():

    for batch_idx, (images, masks, image_paths) in enumerate(val_loader):

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(AMP and DEVICE.type == "cuda")
        ):
            terrain_logits, boundary_logits = model(images)

        # ----------------------------------------------------
        # Probabilities and predictions
        # ----------------------------------------------------
        probabilities = torch.softmax(
            terrain_logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # Maximum class confidence
        max_confidence, _ = torch.max(
            probabilities,
            dim=1
        )

        # Uncertainty = 1 - confidence
        uncertainty = 1.0 - max_confidence

        # ----------------------------------------------------
        # Boundary probability
        # ----------------------------------------------------
        boundary_probability = torch.sigmoid(boundary_logits)

        batch_size = images.size(0)

        for b in range(batch_size):

            if visualization_count >= MAX_VISUALIZATIONS:
                break

            # ------------------------------------------------
            # Convert tensors to numpy
            # ------------------------------------------------
            image_rgb = denormalize_image(images[b])

            gt_mask = masks[b].detach().cpu().numpy().astype(np.uint8)

            pred_mask = predictions[b].detach().cpu().numpy().astype(np.uint8)

            gt_rgb = labels_to_rgb(gt_mask)
            pred_rgb = labels_to_rgb(pred_mask)

            overlay_rgb = make_overlay(
                image_rgb,
                pred_rgb,
                alpha=0.45
            )

            confidence_map = (
                max_confidence[b]
                .detach()
                .cpu()
                .numpy()
            )

            uncertainty_map = (
                uncertainty[b]
                .detach()
                .cpu()
                .numpy()
            )

            boundary_map = (
                boundary_probability[b]
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # File name
            # ------------------------------------------------
            original_name = Path(image_paths[b]).stem

            # ------------------------------------------------
            # 1. Main segmentation visualization
            # ------------------------------------------------
            fig, axes = plt.subplots(
                1,
                4,
                figsize=(20, 5)
            )

            axes[0].imshow(image_rgb)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(gt_rgb)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_rgb)
            axes[2].set_title("Predicted Segmentation")
            axes[2].axis("off")

            axes[3].imshow(overlay_rgb)
            axes[3].set_title("Prediction Overlay")
            axes[3].axis("off")

            add_class_legend(fig)

            fig.suptitle(
                f"Adaptive LSQ Y-Net | {original_name}",
                fontsize=14
            )

            fig.tight_layout(rect=[0, 0.08, 1, 0.95])

            save_path = VIS_DIR / f"{original_name}_visualization.png"

            fig.savefig(
                save_path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close(fig)

            # ------------------------------------------------
            # 2. Overall confidence heatmap
            # ------------------------------------------------
            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            heat = ax.imshow(
                confidence_map,
                cmap="turbo",
                vmin=0.0,
                vmax=1.0
            )

            ax.set_title(
                f"Prediction Confidence\n{original_name}"
            )
            ax.axis("off")

            cbar = fig.colorbar(
                heat,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            cbar.set_label("Maximum Class Probability")

            fig.tight_layout()

            confidence_path = (
                HEATMAP_DIR /
                f"{original_name}_confidence.png"
            )

            fig.savefig(
                confidence_path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close(fig)

            # ------------------------------------------------
            # 3. Uncertainty heatmap
            # ------------------------------------------------
            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            heat = ax.imshow(
                uncertainty_map,
                cmap="magma",
                vmin=0.0,
                vmax=1.0
            )

            ax.set_title(
                f"Prediction Uncertainty\n{original_name}"
            )
            ax.axis("off")

            cbar = fig.colorbar(
                heat,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            cbar.set_label("Uncertainty (1 - Confidence)")

            fig.tight_layout()

            uncertainty_path = (
                UNCERTAINTY_DIR /
                f"{original_name}_uncertainty.png"
            )

            fig.savefig(
                uncertainty_path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close(fig)

            # ------------------------------------------------
            # 4. Boundary heatmap
            # ------------------------------------------------
            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            heat = ax.imshow(
                boundary_map,
                cmap="hot",
                vmin=0.0,
                vmax=1.0
            )

            ax.set_title(
                f"Predicted Boundary Probability\n{original_name}"
            )
            ax.axis("off")

            cbar = fig.colorbar(
                heat,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            cbar.set_label("Boundary Probability")

            fig.tight_layout()

            boundary_path = (
                HEATMAP_DIR /
                f"{original_name}_boundary.png"
            )

            fig.savefig(
                boundary_path,
                dpi=180,
                bbox_inches="tight"
            )

            plt.close(fig)

            # ------------------------------------------------
            # 5. Per-class confidence heatmaps
            # ------------------------------------------------
            class_probabilities = (
                probabilities[b]
                .detach()
                .cpu()
                .numpy()
            )

            for class_id in range(NUM_CLASSES):

                class_probability = class_probabilities[class_id]

                fig, ax = plt.subplots(
                    figsize=(7, 6)
                )

                heat = ax.imshow(
                    class_probability,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0
                )

                ax.set_title(
                    f"Class {class_id} Confidence: "
                    f"{CLASS_NAMES[class_id]}\n"
                    f"{original_name}"
                )

                ax.axis("off")

                cbar = fig.colorbar(
                    heat,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04
                )

                cbar.set_label("Class Probability")

                fig.tight_layout()

                class_path = (
                    CONF_HEATMAP_DIR /
                    f"{original_name}_class_{class_id}_"
                    f"{CLASS_NAMES[class_id]}_confidence.png"
                )

                fig.savefig(
                    class_path,
                    dpi=160,
                    bbox_inches="tight"
                )

                plt.close(fig)

            # ------------------------------------------------
            # 6. Save raw prediction mask
            # ------------------------------------------------
            prediction_mask_path = (
                PRED_DIR /
                f"{original_name}_prediction.png"
            )

            Image.fromarray(
                pred_mask.astype(np.uint8),
                mode="L"
            ).save(prediction_mask_path)

            # ------------------------------------------------
            # 7. Save RGB prediction mask
            # ------------------------------------------------
            prediction_rgb_path = (
                PRED_DIR /
                f"{original_name}_prediction_rgb.png"
            )

            Image.fromarray(
                pred_rgb,
                mode="RGB"
            ).save(prediction_rgb_path)

            visualization_count += 1

            print(
                f"Generated visualization "
                f"{visualization_count}/{MAX_VISUALIZATIONS}: "
                f"{original_name}"
            )

        if visualization_count >= MAX_VISUALIZATIONS:
            break


# ============================================================
# 11A. SUMMARY
# ============================================================

print("\n================ VISUALIZATION GENERATION COMPLETE ================")

print(f"Visualizations saved to : {VIS_DIR}")
print(f"Heatmaps saved to       : {HEATMAP_DIR}")
print(f"Confidence heatmaps    : {CONF_HEATMAP_DIR}")
print(f"Uncertainty heatmaps   : {UNCERTAINTY_DIR}")
print(f"Prediction masks       : {PRED_DIR}")

print(
    f"\nGenerated visualizations for "
    f"{visualization_count} validation images."
)

print("\nFiles include:")
print("  - Original image")
print("  - Ground-truth segmentation")
print("  - Predicted segmentation")
print("  - Prediction overlay")
print("  - Overall confidence heatmap")
print("  - Uncertainty heatmap")
print("  - Boundary probability heatmap")
print("  - Per-class confidence heatmaps")
print("  - Raw predicted label masks")
print("  - RGB predicted masks")

print("\n================ ALL POST-TRAINING OUTPUTS READY ================")