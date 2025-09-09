metadata_file="/data/kalinplus/Cache4Diffusion/eval/geneval/prompts/evaluation_metadata.jsonl"
model_path="/data/public/model/Qwen/Qwen-Image"
outdir="/data/kalinplus/Cache4Diffusion/qwen_image/taylorseer_qwen_image/eval/geneval_outputs"
export CUDA_VISIBLE_DEVICES='0, 2, 4, 5'

# TODO: try different --steps, 20, 25, 30, 50... 
# QwenImagePipeline use EulerDiscreteScheduler by default, 
# but technique report use DPM++ 2M Karras with 20 steps
python qwen_image/taylorseer_qwen_image/eval/qwen_geneval.py \
  --metadata_file "$metadata_file" \
  --model "$model_path" \
  --outdir "$outdir" \
  --n_samples 1 \
  --steps 25 \
  --H 1328 \
  --W 1328 \
  --scale 7.5 \
  --dtype bfloat16 \
  --use_taylor \