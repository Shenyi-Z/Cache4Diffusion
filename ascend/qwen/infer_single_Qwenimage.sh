conda activate modelscope

python qwen_taylorseer_infer.py \
  --prompt "A beautiful painting of a sunset over a calm ocean, with a small boat in the foreground and soft warm lighting." \
  --model Qwen/QwenImage-1.5 \
  --steps 50 \
  --guidance_scale 7.5 \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --dtype fp16 \
  --device npu \
  --enable_taylorseer \
  --outdir outputs \
  --prefix qwen_taylorseer
