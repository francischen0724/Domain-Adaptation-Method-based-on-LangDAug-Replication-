#!/bin/bash
#SBATCH --job-name=EBM_prostate_all
#SBATCH --output=logs/ebm_prostate_all_%j.out
#SBATCH --error=logs/ebm_prostate_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1

set -euo pipefail

ROOT=$(pwd)
DATA_ROOT=./datasets
CHECKPOINT=logs/prostate-BIDMC-BMC-RUNMC-UCL-HK-I2CVB-embed32-nembed256-noisein0.03-LAB/checkpoint/vqvae_best.pt

source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd $ROOT/VQ-VAE

DOMAINS=(BIDMC_train BMC_train RUNMC_train UCL_train HK_train I2CVB_train)
N_SAMPLES=40000

for SOURCE in "${DOMAINS[@]}"; do
  for TARGET in "${DOMAINS[@]}"; do
    if [ "$SOURCE" != "$TARGET" ]; then
      ITER=$((N_SAMPLES / 8))
      echo "🚀 Training EBM: Source $SOURCE → Target $TARGET | N_SAMPLES=$N_SAMPLES | Iters=$ITER"

      python adapt_fundus_prostate_LAB_LD.py \
        --n_gpu 1 \
        --dataset prostate \
        --source $SOURCE \
        --target $TARGET \
        --suffix BIDMC-BMC-RUNMC-UCL-HK-I2CVB \
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
        --n_samples $N_SAMPLES \
        --ae_ckpt $CHECKPOINT \
        --data_root $DATA_ROOT \
        > logs/ebm_${SOURCE}_to_${TARGET}.log 2>&1
    fi
  done
done

echo "✅ All EBM training finished."
