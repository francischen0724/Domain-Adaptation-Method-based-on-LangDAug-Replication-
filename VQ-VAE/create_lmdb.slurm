#!/bin/bash
#SBATCH --job-name=build_lmdb_prostate
#SBATCH --output=logs/build_lmdb_%j.out
#SBATCH --error=logs/build_lmdb_%j.err
#SBATCH --partition=gpu        # 如果不需要GPU，可以改成 cpu 或者短时分区
#SBATCH --gres=gpu:0           # 不用GPU就写0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -e

# 你的环境加载
source /home1/xiwenc/envs/langdaug/bin/activate

# 数据路径
DATA_ROOT=/project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE/datasets/prostate

# 运行
python create_original_lmdb.py \
    --root $DATA_ROOT \
    --splits train,val,test \
    --map_size_gb 128 \
    --commit_every 4096
