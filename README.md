# Domain-Adaptation-Method-based-on-LangDAug-Replication-
Plug and play with the datasets of 2D Retinal fundus and 3D Prostate, find a general way to enhance downstream task (segmentation) performance by dealing with data from different domains.

# 🔬 LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://openreview.net/forum?id=LB5F02kwAv)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/)

<p align="center">
  <img src="teaser.jpg" alt="LangDAug Banner" width="100%">
</p>

## 🌟 Highlights

- **🚀 Boosts Segmentation Accuracy**: Achieves state-of-the-art **87.61% mDSC** (Fundus) & **89.16% DSC** (Prostate MRI).
- **🌉 Bridges Domain Gaps**: First to use Langevin dynamics for domain-bridging data augmentation in segmentation.
- **➕ Enhances Existing Methods**: Improves domain randomization techniques by up to **3.05% mIoU**.
- **💡 Theoretically Sound**: Features proven regularization effects with Rademacher complexity bounds.
- **🩺 Tailored for Medical Imaging**: Specifically designed to overcome challenges in medical image segmentation.

## 📋 Table of Contents

- [Overview](#-overview)
- [Method](#-method)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🔍 Overview

Medical image segmentation models often face a critical challenge: **domain shift**. Performance can significantly degrade when models encounter images from new scanners, hospitals, or patient populations not represented in the training data. This limits their reliability in real-world clinical settings.

**LangDAug (Langevin Data Augmentation)** is a novel technique designed to address this multi-source domain generalization problem in medical image segmentation. Instead of just training on existing source domains, LangDAug intelligently generates new data samples that act as bridges *between* these domains.

It achieves this by:
1.  Training Energy-Based Models (EBMs) to understand the characteristics of different source domains.
2.  Employing Langevin dynamics to sample from these EBMs, creating images that smoothly transition between learned domains.
3.  Augmenting the training data for segmentation models with these new "bridge" samples.

This approach not only improves generalization to unseen domains but also enhances the robustness of the segmentation model.

### ✨ Key Advantages

- **Improved Generalization**: Significantly boosts performance on unseen target domains.
- **Novel Domain Bridging**: Leverages EBMs and Langevin dynamics to create meaningful intermediate domain samples.
- **Seamless Integration**: Can be used as a plug-and-play module to enhance existing domain randomization methods.
- **Strong Theoretical Backing**: The regularization effects are theoretically grounded, ensuring reliable improvements.

## 🔧 Method

Our approach consists of three main steps:

1. **EBM Training**: Train Energy-Based Models to traverse between source domains using contrastive divergence.
2. **Langevin Sampling**: Generate intermediate domain samples through Langevin dynamics.
3. **Augmented Training**: Train segmentation models with both original and Langevin samples.

The overall flow can be visualized as:
```text
[EBM Training] ➔ [Langevin Sampling] ➔ [Augmented Training]
```

## 📊 Results

### Retinal Fundus Segmentation

| Method | Domain A | Domain B | Domain C | Domain D | Avg mIoU | Avg mDSC |
|--------|----------|----------|----------|----------|----------|----------|
| Hutchinson | 66.73 | 66.73 | 69.36 | 66.73 | 67.39 | 78.14 |
| MixStyle | 80.76 | 67.69 | 79.79 | 77.09 | 76.33 | 85.58 |
| FedDG | 76.65 | 72.14 | 76.10 | 75.96 | 75.21 | 83.67 |
| RAM | 77.42 | 73.79 | 79.66 | 78.74 | 77.40 | 85.39 |
| TriD | 80.92 | 72.45 | 79.34 | 78.96 | 77.92 | 85.95 |
| **LangDAug (Ours)** | **78.79** | **75.05** | **81.01** | **80.51** | **78.84** | **87.61** |

### Prostate MRI Segmentation

| Method | Domain A | Domain B | Domain C | Domain D | Domain E | Domain F | Avg ASD | Avg DSC |
|--------|----------|----------|----------|----------|----------|----------|---------|---------|
| Hutchinson | 3.28 | 1.48 | 2.07 | 3.98 | 2.78 | 1.64 | 2.54 | 78.62 |
| MixStyle | 0.72 | 0.88 | 1.62 | 0.65 | 1.59 | 0.51 | 1.00 | 86.27 |
| FedDG | 1.09 | 0.93 | 1.31 | 0.88 | 1.73 | 0.50 | 1.07 | 85.95 |
| RAM | 0.93 | 0.98 | 1.26 | 0.74 | 1.78 | 0.32 | 1.00 | 87.02 |
| TriD | 0.70 | 0.72 | 1.39 | 0.71 | 1.43 | 0.46 | 0.90 | 87.68 |
| **LangDAug (Ours)** | **0.58** | **0.64** | **1.21** | **0.57** | **1.49** | **0.38** | **0.81** | **89.16** |

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.9+

### Setup

1. Clone the repository:

```bash
git clone https://github.com/backpropagator/LangDAug.git
cd LangDAug
```

2. Create and activate conda environment:

```bash
conda env create -f pt-gpu.yml
conda activate pt-gpu
```

## 🎯 Quick Start

### 1. Train VQ-VAE

```bash
bash run_vqvae.sh
```

### 2. Train EBMs for Domain Translation

```bash
bash run_ebm_LAB_LD.sh
```

### 3. Generate Augmented Samples

```bash
bash run_convert_LAB_LD.sh
```

### 4. Train Segmentation Model

```bash
cd ../pytorch-deeplab-xception/
bash train_fundus.sh
```

## 📂 Datasets

### Download Links

- **Fundus Dataset**: [Download](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view)
- **Prostate Dataset**: [Download](https://liuquande.github.io/SAML/)

### Directory Structure
```plaintext
VQ-VAE/
├── datasets/
│   ├── fundus/
│   │   ├── train/
│   │   │   ├── Domain1/
│   │   │   │   ├── image/
│   │   │   │   └── mask/
│   │   │   └── Domain2/
│   │   └── test/
│   └── prostate/
│       ├── BMC/
│       ├── BIDMC/
│       └── ...
```

## 🏋️ Training

### Training with LangDAug

For fundus segmentation:
```bash
python train.py --backbone resnet34 --dataset fundus --splitid 1 2 3 --testid 4 --with_LAB_LD
```
For prostate segmentation:
```bash
python train.py --backbone resnet34 --dataset prostate --splitid BMC RUNMC UCL --testid BIDMC --with_LAB_LD
```
### Training with Domain Randomization + LangDAug

```bash
python train.py --backbone resnet34 --dataset fundus --method TriD --with_LAB_LD
```

## 📈 Evaluation

To evaluate a trained model:
```bash
python train.py --backbone resnet34 --dataset fundus --testid 4 --resume /path/to/checkpoint.pth.tar --testing
```
## 🔬 Ablation Studies

<details>
<summary>Click to expand ablation study results</summary>

### Effect of Langevin Steps (K)

| K | 20 | 40 | 60 | 80 |
|---|----|----|----|----|
| mIoU | 76.2 | 78.8 | 78.5 | 78.3 |

### Effect of Step Size (β)

| β | 0.1 | 0.5 | 1.0 | 2.0 |
|---|-----|-----|-----|-----|
| mIoU | 77.1 | 78.2 | 78.8 | 76.9 |

</details>

## 📖 Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{tiwary2025langdaug,
  title={LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation},
  author={Tiwary, Piyush and Bhattacharyya, Kinjawl and Prathosh, A.P.},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

## 🙏 Acknowledgments

This work was supported by:
- Indian Institute of Science startup grant
- Prime Minister's Research Fellowship
- Kotak IISc AI-ML Center (KIAC)
- Infosys Foundation Young Investigator Award

We thank the authors of [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) and [latent-energy-transport](https://github.com/YangNaruto/latent-energy-transport) for their open-source implementations.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <a href="https://github.com/backpropagator/LangDAug">
    <img src="https://img.shields.io/github/stars/backpropagator/LangDAug?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/backpropagator/LangDAug/issues">
    <img src="https://img.shields.io/github/issues/backpropagator/LangDAug" alt="GitHub issues">
  </a>
  <a href="https://github.com/backpropagator/LangDAug/network">
    <img src="https://img.shields.io/github/forks/backpropagator/LangDAug?style=social" alt="GitHub forks">
  </a>
</p>

<p align="center">
  Made with ❤️
</p>