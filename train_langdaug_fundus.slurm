#!/bin/bash
#SBATCH --job-name=LangDAug_Fundus
#SBATCH --output=logs/fundus_langdaug_%j.out
#SBATCH --error=logs/fundus_langdaug_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1

set -e  # Stop when error

#====================#
# Settings
#====================#
ROOT=$(pwd)
DATASET_SUFFIX='234'
TRAIN_SPLIT="2 3 4"
TEST_SPLIT=1
BACKBONE=resnet34
EXPNAME=fundus_langdaug_BplusCplusD_testA
CHECKPOINT_DIR=$ROOT/pytorch-deeplab-xception/runs/$EXPNAME
LOGDIR=$ROOT/logs

mkdir -p $LOGDIR

#====================#
# 环境激活
#====================#
source /home1/xiwenc/envs/langdaug/bin/activate
export CUDA_VISIBLE_DEVICES=0

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

# #====================#
# # Step 2: Train EBM
# #====================#
# echo "==> Step 2: Train EBM"
# python ebm.py \
#     --dataset fundus \
#     --suffix $DATASET_SUFFIX \
#     --embed_dim 32 \
#     --n_embed 256 \
#     --color_space LAB \
#     --lr 0.001 \
#     --langevin_step 40 \
#     --langevin_lr 1.0 \
#     --batch_size 8 \
#     --ae_ckpt /path/to/vqvae_best.pt \
#     --num_save 5


# #====================#
# # Step 3: Generate LAB-LD converted samples
# #====================#
# echo "==> Step 3: Convert LAB LD images"
# python convert_LAB.py \
#     --dataset fundus \
#     --suffix $DATASET_SUFFIX

# #====================#
# # Step 4: Train segmentation model
# #====================#
# cd $ROOT/pytorch-deeplab-xception
# echo "==> Step 4: Train segmentation model with LangDAug"
# python train.py \
#     --backbone $BACKBONE \
#     --dataset fundus \
#     --splitid $TRAIN_SPLIT \
#     --testid $TEST_SPLIT \
#     --with_LAB_LD

# #====================#
# # Step 5: Evaluate on all domains (optional)
# #====================#
# echo "==> Step 5: Evaluate on domains A/B/C/D"
# CHECKPOINT=$(ls $CHECKPOINT_DIR/*.pth* 2>/dev/null | head -n 1)

# for TID in 1 2 3 4; do
#   echo "---- Evaluating on Domain $TID ----"
#   if [ -n "$CHECKPOINT" ]; then
#     python train.py \
#       --backbone $BACKBONE \
#       --dataset fundus \
#       --testid $TID \
#       --resume $CHECKPOINT \
#       --with_LAB_LD
#   else
#     echo "[WARNING] No checkpoint found to evaluate on Domain $TID"
#   fi
# done

# echo "✅ DONE ALL"
