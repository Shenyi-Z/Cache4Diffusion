conda activate modelscope

python3 batch_infer.py \
    --model ./qwenimage \
    --steps 50 \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --dtype bfloat16 \
    --true_cfg_scale 7.5 \
    --outdir outputs \
    --prefix QwenImage \
    --prompt_file sample_prompts.txt \
    --enable_taylorseer
