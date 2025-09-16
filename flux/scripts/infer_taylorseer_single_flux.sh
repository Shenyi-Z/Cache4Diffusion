cd flux
python taylorseer_flux/diffusers_taylorseer_flux.py \
    --model black-forest-labs/FLUX.1-dev \
    --steps 50 \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --dtype float16 \
    --guidance_scale 7.5 \
    --outdir outputs \
    --prefix TaylorSeer \
    --prompt "A beautiful painting of a sunset over a calm ocean, with a small boat in the foreground, and a few clouds in the sky. The colors are warm and the lighting is soft, creating a serene and peaceful atmosphere. The painting is detailed and the brush strokes are visible, adding to the realism of the scene." \
