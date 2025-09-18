#!/bin/bash
#SBATCH --job-name=Train_Fundus_Seg_A_all
#SBATCH --output=logs/train_fundus_seg_A_all_%j.out
#SBATCH --error=logs/train_fundus_seg_A_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

set -e

ROOT=$(pwd)
DATA_ROOT=/project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE/datasets/fundus


source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

cd $ROOT/pytorch-deeplab-xception

python train.py \
    --backbone resnet34 \
    --loss-type bce-dice \
    --optimizer AdamW \
    --lr 1e-4 \
    --momentum 0.9 \
    --workers 8 \
    --epochs 160 \
    --batch-size 8 \
    --base-size 256 \
    --crop-size 256 \
    --gpu-ids 0 \
    --checkname deeplab-resnet34 \
    --eval-interval 1 \
    --dataset fundus \
    --data_root $DATA_ROOT \
    --splitid 1 2 3 4 \
    --valid 1 \
    --testid 1 \
    --with_LAB_LD
