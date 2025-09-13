metadata_file="/data/huangkailin-20250908/eval/geneval/prompts/evaluation_metadata.jsonl"
model_path="/data/public/models/Qwen/Qwen-Image"
outdir="/data/huangkailin-20250908/eval/outputs/geneval/origin"
# seems useless in tmux
export CUDA_VISIBLE_DEVICES='7'

# TODO: try different --steps, 20, 25, 30, 50... 
# QwenImagePipeline use EulerDiscreteScheduler by default, 
# but technique report use DPM++ 2M Karras with 20 steps
python qwen_image/eval/qwen_geneval.py \
  --metadata_file "$metadata_file" \
  --model "$model_path" \
  --outdir "$outdir" \
  --n_samples 1 \
  --steps 50 \
  --H 1328 \
  --W 1328 \
  --scale 7.5 \
  --dtype bfloat16 \
  # --use_taylor \