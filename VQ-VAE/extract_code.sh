#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python extract_code.py --ckpt /provide/path \
	--name provide_name \
	--data_path /provide/path \