cluster_num=32
propagation_ratio=0.01
threshold=5
python sample_video.py \
    --video-size 480 640 \
    --video-length 65 \
    --infer-steps 50 \
    --flow-reverse \
    --use-cpu-offload \
    --seed 42 \
    --save-path "./" \
    --prompt "A cat walks on the grass, realistic style." \
    --mode ClusCa \
    --max-order 1 \
    --fresh-threshold $threshold \
    --propagation-ratio $propagation_ratio \
    --cluster-num $cluster_num \
    --k 1