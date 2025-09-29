#!/bin/bash

# Define an array of prompts
prompts=(
       "A person is rock climbing"
)

# Loop through each prompt and run the script
for prompt in "${prompts[@]}"; do
    python3 sample_video.py \
        --video-size 480 640 \
        --video-length 65 \
        --infer-steps 50 \
        --prompt "$prompt" \
        --seed 42 \
        --embedded-cfg-scale 6.0 \
        --flow-shift 7.0 \
        --flow-reverse \
        --use-cpu-offload \
        --save-path ./results/$(echo "$prompt" | tr -s ' ' '_')
done
