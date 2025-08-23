cd flux
python taylorseer_flux/batch_infer.py \
    --model black-forest-labs/FLUX.1-dev \
    --steps 50 \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --dtype float16 \
    --guidance_scale 7.5 \
    --outdir outputs \
    --prefix TaylorSeer \
    --prompt_file prompts.txt \