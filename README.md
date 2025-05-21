# LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation

This repository contains the official implementation of our method LangDAug. It is designed to perform domain generalization on medical imaging datasets (tested on prostate and fundus) through data augmentation using Energy-Based Models (EBMs).

## Requirements

To install the necessary dependencies, we recommend setting up a Conda environment.

```bash
conda env create -f pt-gpu.yml
conda activate pt-gpu
```

## Code Structure

- **VQ-VAE/**: Scripts required to train the VQ-VAE, the EBMs operating on the VQ-VAE latent space, and to generate translated images (augmentations) from one domain to another. Also contains the `datasets` folder.

- **pytorch-deeplab-xception/**: Scripts to train the segmentation models.

## Datasets

- **Prostate Dataset**: [Link](https://liuquande.github.io/SAML/)  
- **Fundus Dataset**: [Link](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view)

### Directory Structure

Place the datasets in the following structure:

```plaintext
VQ-VAE/
├── datasets/
│   ├── fundus/
│   │   ├── train/
│   │   │   ├── Domain1/
│   │   │   │   ├── image/
│   │   │   │   └── mask/
│   │   │   └── Domain2/
│   │   │       ├── image/
│   │   │       └── mask/
│   │   └── test/
│   │       ├── Domain1/
│   │       │   ├── image/
│   │       │   └── mask/
│   │       └── Domain2/
│   │           ├── image/
│   │           └── mask/
│   └── prostate/
│       ├── BMC/
│       ├── BIDMC/
│       └── OtherDomains/
```
For running the prostate trainings faster, it is highly recommended to create an LMDB folder from the original dataset as well as for the translated samples. This can be done using the `create_lmdb.py` file in the `VQ-VAE/` folder.

## Data Augmentation

- **Training the VQ-VAE**: Use the `run_ae.sh` script in the `VQ-VAE/` folder.

- **Training EBMs for Domain Translation**: To train the Energy-Based Models (EBMs) for translating between pairs of domains, use the `run_ebm_LAB_LD.sh` script in the `VQ-VAE/` folder.

- **Generating Translations Between Domains**: To obtain translations from one domain to another, use the `run_convert_LAB_LD.sh` script in the `VQ-VAE/` folder.

## Training and Evaluation

- **Training the Segmentation Model**: Use the `train_fundus.sh` script in the `pytorch-deeplab-xception/` folder.

- **Testing Trained Checkpoints**: To test the trained checkpoints, use the `test.sh` script in the `pytorch-deeplab-xception/` folder.

## Acknowledgments

This repository builds upon code from [deeplab](https://github.com/jfzhang95/pytorch-deeplab-xception) and [latent-energy-transport](https://github.com/YangNaruto/latent-energy-transport). We are grateful to the authors for sharing their work and making it publicly accessible.


