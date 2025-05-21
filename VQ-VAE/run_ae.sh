#!/bin/bash
DATASET='123'
export CUDA_VISIBLE_DEVICES=0

python train_vqvae_fundus_prostate.py --data_path ./datasets/prostate/ \
	--n_gpu 1 --batch_size 64 --dataset prostate --suffix $DATASET --domains BIDMC BMC HK I2CVB RUNMC UCL \
	--size 384 --embed_dim 64 --n_embed 512 --input_noise 0.03 --color_space LAB \