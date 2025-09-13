#!/bin/bash

# Define an array of prompts
prompts=(
    "In a still frame, a stop sign"
    "In a still frame, the Parthenon's majestic Doric columns stand in serene solitude atop the Acropolis, framed by the tranquil Athenian landscape"
    "an elephant and a bear"
    "a motorcycle accelerating to gain speed"
    "a cat drinking water"
    "an elephant running to join a herd of its kind"
    "a bed"
    "a sink"
    "A tranquil tableau of bedroom"
    "A tranquil tableau of indoor library"
    "a yellow umbrella"
    "a toaster and a teddy bear"
    "an oven and scissors"
    "a sandwich and a book"
    "a hair drier and a toothbrush"
    "airplane and train"
    "an elephant spraying itself with water using its trunk to cool down"
    "a motorcycle"
    "a tie"
    "sports ball and kite"
    "an airplane accelerating to gain speed"
    "a dog drinking water"
    "a dog playing in park"
     "a surfboard on the bottom of skis, front view"
      "a skateboard on the bottom of a surfboard, front view"
      "a kite on the bottom of a skateboard, front view"
      "supermarket"
      "staircase"
      "skyscraper"
      "downtown"
      "corridor"
      "Snow rocky mountains peaks canyon. snow blanketed rocky mountains surround and shadow deep canyons. the canyons twist and bend through the high elevated mountain peaks, zoom in"
      "An astronaut flying in space, with an intense shaking effect"
      "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, with an intense shaking effect"
      "a purple bird"
      "a yellow bird"
      "a green bird"
      "a white car"
       "a purple car"
       "a yellow car"
       "a black bicycle"
       "A person is trimming or shaving beard"
       "A person is ironing"
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
