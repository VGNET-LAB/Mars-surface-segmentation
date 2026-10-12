
MarTA-Net:  Terrain-Aware U-Net Enhanced with Convolutional Block Attention Modules for Martian Surface Segmentation

Automated and accurate segmentation of Martian terrain is crucial for planetary science, as it enables safe rover navigation, facilitates geological analysis, and informs landing-site selection. Deep learning models, particularly U-Net architectures, have shown promise in this domain. However, standard segmentation models can be improved by focusing on more salient features within the complex and subtle Martian landscape. This paper introduces MarTA-Net, a novel semantic segmentation model that integrates the Convolutional Block Attention Mechanism (CBAM) into a U-Net architecture. By sequentially applying channel and spatial attention modules, MarTA-Net refines feature maps at each stage of the encoder and decoder, enabling the model to focus on the most informative features in terms of "what" and "where". We train and evaluate the proposed model on a curated dataset of Martian orbital images, featuring classes such as craters, rough terrain, smooth terrain, and alluvial fans. To address class imbalance, particularly for the rare but scientifically significant alluvial fan class, we employ a combined loss function of weighted Cross-Entropy and Dice loss. To test MarTA-Net's generalization, the S5Mars dataset was used. Experimental results show that MarTA-Net outperforms baseline models across key performance metrics, demonstrating its effectiveness in producing more precise Martian terrain maps. Additionally, to support deployment in resource-constrained environments, a quantized student model was developed that significantly reduces computational requirements, enabling implementation on edge devices for further onboard planetary analysis. 

The framework is designed for adaptive rover landing surface prediction and large-scale Martian terrain understanding. The model categorizes terrain into multiple geological classes, including:

- Extremely Smooth Terrain  
- Smooth Terrain  
- Rough Terrain  
- Extremely Rough Terrain  
- Craters  
- Alluvial Fans  

To address the challenge of insufficient labeled data, a **self-updating retraining framework** is incorporated to continuously improve segmentation performance using iterative pseudo-label refinement. Additionally, to mitigate class imbalance — especially for rare but scientifically significant classes such as alluvial fans — a combined loss function consisting of **Weighted Cross-Entropy Loss** and **Dice Loss** is employed.

The model is trained and evaluated using high-resolution Martian orbital imagery obtained from the **Mars Orbital Data Explorer (ODE)**, including **CTX** and **HiRISE** datasets. Generalization capability is further evaluated using the **S5Mars dataset**. Extensive experiments compare the proposed architecture against several state-of-the-art segmentation models, including:

- U-Net  
- DeepLabV3+  
- DPT  
- SegFormer  

Performance evaluation is conducted using standard semantic segmentation metrics such as:

- Intersection over Union (IoU)  
- Dice Coefficient  
- Precision  
- Recall  

Experimental results demonstrate that the proposed attention-enhanced architecture achieves improved segmentation accuracy and more precise Martian terrain mapping compared to baseline approaches. Furthermore, to support deployment in computationally constrained environments, a **quantized lightweight student model** is developed for efficient onboard inference and edge-device deployment in future autonomous planetary missions.

## Repository Contents

This repository contains:

- Model architectures  
- Training and evaluation pipelines  
- Dataset preprocessing scripts  
- Attention-based U-Net implementations  
- Quantization workflows  
- Experimental notebooks  
- Benchmarking and comparison code  

## Objective

The project aims to contribute toward robust AI-driven planetary surface analysis and safer autonomous exploration systems for future Mars missions.

