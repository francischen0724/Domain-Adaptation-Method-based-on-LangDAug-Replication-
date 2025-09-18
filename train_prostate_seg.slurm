#!/bin/bash
#SBATCH --job-name=Train_prostate_Seg_Reverse_UCL
#SBATCH --output=logs/train_prostate_seg_Reverse_UCL_%j.out
#SBATCH --error=logs/train_prostate_seg_Reverse_UCL_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=32:00:00
#SBATCH --ntasks=1

set -e

ROOT=$(pwd)
DATA_ROOT=/project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE/datasets/prostate


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
    --epochs 100 \
    --batch-size 32 \
    --base-size 384 \
    --crop-size 384 \
    --gpu-ids 0 \
    --checkname deeplab-resnet34 \
    --eval-interval 1 \
    --dataset prostate \
    --data_root $DATA_ROOT \
    --splitid UCL 126 621 226 622 326 623 426 624 526 625 \
    --valid UCL \
    --testid UCL \
    --with_LAB_LD
