local_model_path="/data/public/model/Qwen/Qwen-Image"
model_id="Qwen/Qwen-Image"
export CUDA_VISIBLE_DEVICES='5, 6'

python qwen_image/taylorseer_qwen_image/batch_infer.py \
    --model "$local_model_path" \
    --steps 50 \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --dtype bfloat16 \
    --true_cfg_scale 7.5 \
    --outdir outputs/DrawBench200/taylor \
    --prefix TaylorSeer \
    --prompt_file assets/prompts/DrawBench200.txt \