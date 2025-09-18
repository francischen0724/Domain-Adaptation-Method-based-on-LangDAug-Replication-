#!/bin/bash
#SBATCH --job-name=Compare_EBM_Checkpoints
#SBATCH --output=logs/compare_ebm_%j.out
#SBATCH --error=logs/compare_ebm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1          # 申请 1 块 GPU
#SBATCH --cpus-per-task=4     # 4 个 CPU core
#SBATCH --mem=32G             # 内存大小
#SBATCH --time=2:00:00        # 最长运行时间

# 加载 CUDA 和 Python 环境
# module load cuda/11.8
# module load python/3.11

# 激活虚拟环境
source ~/envs/langdaug/bin/activate

# 进入代码目录
cd /project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE

# 运行对比脚本
python compare_ebm_checkpoints.py
