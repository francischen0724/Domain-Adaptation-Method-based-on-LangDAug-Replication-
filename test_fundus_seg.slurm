#!/bin/bash
#SBATCH --job-name=Test_Fundus_Seg_A_all
#SBATCH --output=logs/test_fundus_A_all_%j.out
#SBATCH --error=logs/test_fundus_A_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

set -e

ROOT=$(pwd)
DATA_ROOT=/project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE/datasets/fundus
CHECKPOINT=./run/fundus/deeplab-resnet34/experiment_1_2_3_4_10/model_best.pth.tar
source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

cd $ROOT/pytorch-deeplab-xception

for DOMAIN in 1 2 3 4
do
    echo "==== Testing Domain $DOMAIN ===="
    python train.py \
        --backbone resnet34 \
        --loss-type bce-dice \
        --optimizer AdamW \
        --lr 1e-4 \
        --workers 8 \
        --batch-size 1 \
        --dataset fundus \
        --data_root $DATA_ROOT \
        --epochs 5000 \
        --base-size 256 \
        --crop-size 256 \
        --gpu-ids 0 \
        --resume $CHECKPOINT \
        --testing \
        --testid $DOMAIN
done
