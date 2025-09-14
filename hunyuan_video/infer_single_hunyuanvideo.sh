#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

model="/data/public/models/tencent/HunyuanVideo"

python hunyuan_video/taylorseer_hunyuan_video/diffusers_taylorseer_hunyuan_video.py \
    --prompt "A beautiful sunrise over the mountains" \
    --video-length 129 \
    --video-size 720 1280 \
    --fps 30 \
    --infer-steps 50 \
    --model "$model" \
    --use_taylor