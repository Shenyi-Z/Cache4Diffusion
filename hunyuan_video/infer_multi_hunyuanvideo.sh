#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
echo $CUDA_VISIBLE_DEVICES

model="/data/public/models/hunyuanvideo-community/HunyuanVideo"
prompt_file="/data/huangkailin-20250908/Cache4Diffusion/assets/prompts/test.txt"

python hunyuan_video/taylorseer_hunyuan_video/batch_infer.py \
    --prompt_file "$prompt_file" \
    --video-length 19 \
    --video-size 544 960 \
    --fps 4 \
    --infer-steps 50 \
    --model "$model" \
    --use_taylor \
    --dtype float16 \