# local_model_path="/data/public/model/Qwen/Qwen-Image"
model_id="Qwen/Qwen-Image"

python qwen_image/taylorseer_qwen_image/batch_infer.py \
    --model "$model_id" \
    --steps 50 \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --dtype bfloat16 \
    --true_cfg_scale 7.5 \
    --outdir outputs \
    --prefix TaylorSeer \
    --prompt_file assets/prompts/miku.txt \