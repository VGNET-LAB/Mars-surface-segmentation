# Martian Terrain Segmentation

**VG-NET Lab, IISER Thiruvananthapuram**  
**Research support:** Indian Space Research Organisation (ISRO)

This repository documents the development of computer-vision methods for multi-class semantic segmentation of Martian surface imagery. The work focuses on building, evaluating, and improving models that identify terrain units and geomorphological features in orbital images. The repository also records dataset preparation, model optimization, and evaluation as the research progresses.

> **Project status:** MarTA-Net has been published in the *IEEE Internet of Things Journal*. The curated MarTA-Net dataset is available on Hugging Face. Y-Net and CCHNet are additional model tracks under experimentation.

## Research objectives

The project aims to support automated Martian terrain mapping by:

- Developing semantic-segmentation models for terrain and landform identification.
- Curating and preparing image–mask pairs for multi-class training and evaluation.
- Studying attention mechanisms, boundary refinement, and approaches to class imbalance.
- Investigating model binarization and quantization for more efficient inference.
- Comparing models using reproducible evaluation procedures and class-wise metrics.

These methods are intended for research in planetary image analysis, geological mapping, rover mobility, and landing-site assessment. 

## Current research tracks

| Model / resource | Current status | Summary |
|---|---|---|
| **MarTA-Net** | **Published** | Terrain-aware U-Net enhanced with the Convolutional Block Attention Module (CBAM). The work studies channel and spatial attention for multi-class Martian surface segmentation and includes a lightweight quantized student-model direction. |
| **Y-Net** | Experimental development | Encoder–decoder segmentation network using an EfficientNet-B0 encoder, Adaptive Gated Convolutional Blocks (AGCB), Channel-wise and Spatial Attention Integration Blocks (CSAIB), and Fusion Attention Refinement Blocks (FARB). It has a terrain-segmentation head and a separate boundary-prediction head. Its training pipeline includes rare-class-aware sampling and a composite loss. |
| **CCHNet** | Experimental development | A separate model track with model implementation and binarization/quantization experiments. The relevant scripts and notebook are grouped under `models/CCHNet/`. Architecture details and comparative results should be updated here as experiments are finalized. |
| **MarTA-Net dataset** | Published | Curated Martian surface imagery and segmentation masks released on Hugging Face for research on multi-class terrain segmentation. |

Model status reflects the current repository roadmap; experimental implementations and results may change as evaluation continues.

## Published work and dataset

### MarTA-Net publication

**Chatterjee, Anoushka; Kandi, Kaustubh Anand; Adhikari, Mainak.** “MarTA-Net: A Terrain-Aware U-Net Enhanced with Convolutional Block Attention Modules for Martian Surface Segmentation.” *IEEE Internet of Things Journal*, 2026.

- **IEEE Xplore:** [Read the publication](https://ieeexplore.ieee.org/abstract/document/11664087)
- **DOI:** [10.1109/JIOT.2026.3726564](https://doi.org/10.1109/JIOT.2026.3726564)

### Curated dataset

The updated MarTA-Net dataset is hosted on Hugging Face:

- **Dataset repository:** [Anoushkaacc/MarTA-Net](https://huggingface.co/datasets/Anoushkaacc/MarTA-Net)

Refer to the dataset card for the current structure, label mapping, usage notes, and any updates to these counts.

## Repository structure

```text
.
├── Dataset Preparation Folder/       # Dataset preparation resources
├── models/
│   ├── CCHNet/
│   │   ├── CCHNet_code.py             # CCHNet implementation
│   │   ├── CCHNet_binarization.py     # Binarization experiment
│   │   ├── CCHNet_quantization.py     # Quantization experiment
│   │   ├── Copy_of_BIN_OUTPUT.ipynb   # Binarization-related notebook
│   │   └── del.md
│   ├── ynet/
│   │   ├── ynet-class-weights.ipynb   # Class-weight experiments
│   │   └── ynet-model-trial.ipynb     # Y-Net model experiments
│   └── Dynamic_MARTANET.ipynb         # MarTA-Net experiments
├── validation_metrics.py              # Segmentation evaluation utilities
├── validation_test_images.py          # Evaluation on test images
├── validation_with_gt.py              # Evaluation using ground-truth masks
├── LICENSE
└── README.md
```

This tree reflects the layout currently tracked in the repository. Notebooks and scripts may be added, renamed, or reorganized during development.

## Evaluation and experimentation

The repository includes validation scripts and model-specific experiments. Evaluation report includes both aggregate and class-wise performance wherever possible, including:

- Intersection over Union (IoU) and mean IoU (mIoU)
- Dice coefficient
- Precision and recall
- Boundary quality, where a model predicts terrain boundaries
- Computational measures such as model size, inference time, and throughput for optimization experiments

For reliable comparison, evaluate models with consistent preprocessing, class definitions, data splits, and metric implementations. Where augmented images originate from the same manually annotated source image, use source-grouped train/validation/test splits to reduce augmentation leakage. Record the dataset version, random seed, training configuration, checkpoint, and evaluation protocol with each reported result.

## Research progress log

Updates are recorded here to make the evolution of the project visible. Add dated entries as milestones are completed.

| Date / period | Milestone | Status |
|---|---|---|
| 2025 - 2026 | MarTA-Net paper published in the *IEEE Internet of Things Journal*. | Completed |
| 2026 | Updated MarTA-Net dataset (12 class) released on Hugging Face. | Completed |
| 2026 | Y-Net experimentation includes attention-enhanced encoder–decoder features, a boundary-prediction branch, and rare-class-aware training. | In progress |
| 2026 | CCHNet implementation and associated binarization and quantization scripts are present in the repository. | In progress |
| Ongoing | Standardized evaluation, cross-model comparisons, and documentation of model efficiency and generalization. | Planned / ongoing |

## Reproducibility notes

The repository contains Python scripts and Jupyter notebooks developed at different stages of the project. Before running an experiment:

1. Review the selected script or notebook for expected dataset paths, input dimensions, dependencies, and output locations.
2. Confirm that image files and masks use the expected label encoding and class mapping.
3. Keep augmented variants of the same source image within a single data split.
4. Save configuration files, checkpoints, logs, and metric outputs for each run.
5. Report limitations and per-class performance, especially where classes are rare or visually ambiguous.

A single unified installation and training command is not documented yet because the model tracks may have different requirements and experiment configurations. Please use the instructions embedded in the relevant model files until a consolidated environment specification is added.

## Funding and acknowledgement

This research is conducted by **VG-NET Lab at IISER Thiruvananthapuram** with support from the **Indian Space Research Organisation (ISRO)**. Publications and downstream use of project outputs should acknowledge the applicable funding and cite the relevant publication and dataset version.

## Citation

If you use MarTA-Net or its associated dataset in research, please cite the publication and refer to the dataset repository.

```bibtex
@article{chatterjee2026martanet,
  author  = {Chatterjee, Anoushka and Kandi, Kaustubh Anand and Adhikari, Mainak},
  title   = {MarTA-Net: A Terrain-Aware U-Net Enhanced with Convolutional Block Attention Modules for Martian Surface Segmentation},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  doi     = {10.1109/JIOT.2026.3726564}
}
```

- Paper: https://doi.org/10.1109/JIOT.2026.3726564
- Dataset: https://huggingface.co/datasets/Anoushkaacc/MarTA-Net
- Repository: https://github.com/VGNET-LAB/Mars-surface-segmentation

## License

See [`LICENSE`](LICENSE) for the repository's software license. Check the dataset card and any source-dataset terms separately before redistributing or using the data.
