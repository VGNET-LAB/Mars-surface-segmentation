
#  Mars Surface Segmentation with CBAM-UNet

A comprehensive semantic segmentation pipeline for Mars surface imagery using a Convolutional Block Attention Module (CBAM) – enhanced U‑Net architecture. This repository provides scripts for data preprocessing, model training, evaluation, inference, visualization, and pseudo‑labeling support.
---

## 🧠 Model & Methodology

**Architecture:** U‑Net with integrated CBAM (Channel + Spatial Attention) blocks

**Segmentation Classes:**

* **0 – Background**  `(0, 0, 0)`
* **1 – Crater**      `(255, 255, 0)`
* **2 – Rough**       `(255, 0, 0)`
* **3 – Smooth**      `(0, 255, 0)`
* **4 – Alluvial Fan**`(0, 0, 255)`
* **5 – Boulders**    `(139, 69, 19)`

**Loss Function:** Weighted Cross‑Entropy + Dice Loss (background ignored)

**Training:** PyTorch with mixed‑precision (AMP) and OneCycleLR scheduler

**Evaluation Metrics:** Per-class IoU, Dice, Precision, Recall; Mean IoU; Average inference time

---

---
Contibutors: [Anoushka Chatterjee](https://github.com/anoushkaacc) and [Keerthana A R](https://github.com/KeerthanaKodes)
