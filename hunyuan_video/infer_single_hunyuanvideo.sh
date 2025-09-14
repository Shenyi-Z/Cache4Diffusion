#!/bin/bash

export CUDA_VISIBLE_DEVICES='0,7'
echo $CUDA_VISIBLE_DEVICES

model="/data/public/models/hunyuanvideo-community/HunyuanVideo"

python hunyuan_video/taylorseer_hunyuan_video/diffusers_taylorseer_hunyuan_video.py \
    --prompt "A beautiful sunrise over the mountains" \
    --video-length 19 \
    --video-size 544 960 \
    --fps 4 \
    --infer-steps 50 \
    --model "$model" \
    --use_taylor \
    --dtype bfloat16 \