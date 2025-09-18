#!/bin/bash
#SBATCH --job-name=VQVAE_Prostate_fold_all
#SBATCH --output=logs/vqvae_prostate_BIDMC_BMC_RUNMC_UCL_HK_I2CVB_%j.out
#SBATCH --error=logs/vqvae_prostate_BIDMC_BMC_RUNMC_UCL_HK_I2CVB_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1

set -e  # Stop when error

#====================#
# Settings
#====================#
ROOT=$(pwd)
DOMAINS="BIDMC BMC RUNMC UCL HK I2CVB"
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
echo "==> Train VQ-VAE (prostate) on domains: $DOMAINS"

python train_vqvae_fundus_prostate.py \
    --dataset prostate \
    --data_path ./datasets/prostate \
    --domains $DOMAINS \
    --size 384 \
    --embed_dim 32 \
    --n_embed 256 \
    --batch_size 64 \
    --input_noise 0.03 \
    --color_space LAB \
    --lr 0.001