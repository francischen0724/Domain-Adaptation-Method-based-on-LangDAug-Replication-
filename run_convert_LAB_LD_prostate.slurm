#!/bin/bash
#SBATCH --job-name=Convert_LAB_prostate
#SBATCH --output=logs/convert_lab_prostate_%j.out
#SBATCH --error=logs/convert_lab_prostate_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --ntasks=1

set -e

ROOT=$(pwd)
DATA_ROOT=./datasets
CHECKPOINT_VQ=logs/prostate-BIDMC-BMC-RUNMC-UCL-HK-I2CVB-embed32-nembed256-noisein0.03-LAB/checkpoint/vqvae_best.pt
EBM_CKPT=results/prostate-RUNMC_train2HK_train/00000-ls40-llr1.0000-lr0.0002-embed32-nembed256-attn-sn-chm8-beta0.9_0.999-tgt-HK/ebm_004000.pt

EXPNAME=LAB_LD_prostate
# EXPNAME=LAB_LD_prostate_all

source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd $ROOT/VQ-VAE

python convert_LAB.py \
    --dataset prostate \
    --source RUNMC_train \
    --target HK_train \
    --suffix E2C \
    --channel_mul 8 \
    --langevin_step 40 \
    --langevin_lr 1.0 \
    --lr 0.0002 \
    --beta1 0.9 \
    --beta2 0.999 \
    --attention \
    --n_embed 256 \
    --embed_dim 32 \
    --batch_size 8 \
    --sn \
    --ae_ckpt $CHECKPOINT_VQ \
    --ebm_ckpt $EBM_CKPT \
    --data_root $DATA_ROOT \
    --num_save 39 \
    --color_space LAB \
    --expt_name $EXPNAME
