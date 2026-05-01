# Leukocyte Segmentation using U-Net with Bottleneck Attention and Boundary Loss

## 📌 Overview
This project presents a deep learning framework for automatic segmentation of leukocytes (white blood cells), focusing on cytoplasm and nuclei delineation in bone marrow images.

The proposed method integrates:
- Bottleneck Attention Module (BAM)
- Boundary-Proximal Loss (BPL)
- U-Net with ResNet-34 backbone

It is designed to handle:
- Overlapping cells  
- Complex morphology  
- Blurred boundaries  

---

## 🧠 Architecture
- Model: U-Net  
- Encoder: ResNet-34 (ImageNet pretrained)  
- Attention: Bottleneck Attention Module  
- Loss:
  - Dice Loss (multi-class)
  - Boundary Loss (distance-based)

---

## 📂 Dataset
Bone marrow cytological images with RGB masks converted into 3 classes:

- 0 → Background  
- 1 → Cytoplasm  
- 2 → Nucleus  
