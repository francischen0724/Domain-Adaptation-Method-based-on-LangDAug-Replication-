#!/bin/bash
#SBATCH --job-name=Test_prostate_Seg_BIDMC
#SBATCH --output=logs/test_prostate_BIDMC_%j.out
#SBATCH --error=logs/test_prostate_BIDMC_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

set -e

ROOT=$(pwd)
DATA_ROOT=/project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE/datasets/prostate
CHECKPOINT=./run/prostate/deeplab-resnet34/experiment_BIDMC_16/model_best.pth.tar
source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

cd $ROOT/pytorch-deeplab-xception
DOMAINS=(BIDMC BMC HK I2CVB RUNMC UCL)

for DOM in "${DOMAINS[@]}"; do
    echo "==== Testing Domain ${DOM} (size=384) ===="
    python train.py \
        --backbone resnet34 \
        --loss-type bce-dice \
        --optimizer AdamW \
        --lr 1e-4 \
        --workers 8 \
        --batch-size 1 \
        --dataset prostate \
        --data_root $DATA_ROOT \
        --epochs 5000 \
        --base-size 384 \
        --crop-size 384 \
        --gpu-ids 0 \
        --resume $CHECKPOINT \
        --testing \
        --testid "$DOM"
done
