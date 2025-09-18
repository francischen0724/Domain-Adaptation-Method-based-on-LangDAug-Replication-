#!/bin/bash
#SBATCH --job-name=EBM_Fundus_124
#SBATCH --output=logs/ebm_fundus_124_%j.out
#SBATCH --error=logs/ebm_fundus_124_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1

set -e

ROOT=$(pwd)
DATA_ROOT=./datasets
CHECKPOINT=logs/fundus-124-embed32-nembed256-noisein0.03-LAB/checkpoint/vqvae_best.pt

source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd $ROOT/VQ-VAE

for SOURCE in 1 2 4; do
  for TARGET in 1 2 4; do
    if [ "$SOURCE" != "$TARGET" ]; then
      case $SOURCE in
        1) N_SAMPLES=16000 ;;
        2) N_SAMPLES=16000 ;;
        3) N_SAMPLES=32000 ;;
        4) N_SAMPLES=32000 ;;
      esac
      ITER=$((N_SAMPLES / 8))
      echo "🚀 Training EBM: Source $SOURCE → Target $TARGET | N_SAMPLES=$N_SAMPLES | Iters=$ITER"

      python adapt_fundus_prostate_LAB_LD.py \
        --n_gpu 1 \
        --dataset fundus \
        --source $SOURCE \
        --target $TARGET \
        --suffix 124 \
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
        --data_root $DATA_ROOT
    fi
  done
done

echo "✅ All EBM training finished."
