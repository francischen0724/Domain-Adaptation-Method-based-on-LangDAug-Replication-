#!/bin/bash
#SBATCH --job-name=LangDAug_Fundus_1234
#SBATCH --output=logs/fundus_langdaug_1234_%j.out
#SBATCH --error=logs/fundus_langdaug_1234_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1

set -e  # Stop when error

#====================#
# Settings
#====================#
ROOT=$(pwd)
DATASET_SUFFIX='1234'
# TRAIN_SPLIT="2 3 4"
# TEST_SPLIT=1
# BACKBONE=resnet34
# EXPNAME=fundus_langdaug_BplusCplusD_testA
# EXPNAME=fundus_langdaug_AplusCplusD_testB
# EXPNAME=fundus_langdaug_AplusBplusD_testC
EXPNAME=fundus_langdaug_AplusBplusCplusD

# CHECKPOINT_DIR=$ROOT/pytorch-deeplab-xception/runs/$EXPNAME
LOGDIR=$ROOT/logs

mkdir -p $LOGDIR

#====================#
# 环境激活
#====================#
source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

#====================#
# Step 1: Train VQ-VAE
#====================#
cd $ROOT/VQ-VAE
echo "==> Step 1: Train VQ-VAE"

python train_vqvae_fundus_prostate.py \
    --dataset fundus \
    --data_path ./datasets/fundus \
    --suffix $DATASET_SUFFIX \
    --embed_dim 32 \
    --n_embed 256 \
    --batch_size 64 \
    --input_noise 0.03 \
    --color_space LAB \
    --lr 0.001