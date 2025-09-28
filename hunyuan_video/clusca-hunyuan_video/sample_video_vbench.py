#!/usr/bin/env python3
import os
# First, set environment variables to disable tokenizers warning messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import contextlib
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

@contextlib.contextmanager
def suppress_output():
    """
    Temporarily suppress standard output, standard error, and loguru output.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            # Disable loguru output
            logger.disable("")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.enable("")

def main():
    # Get command-line arguments
    args = parse_args()

    # New parameter: If --vbench-json-path is provided, load the JSON file; otherwise, keep the original logic (single prompt)
    vbench_json_path = getattr(args, "vbench_json_path", None)
    index_start = int(getattr(args, "index_start", 0))
    index_end   = int(getattr(args, "index_end", -1))
    num_videos_per_prompt = int(getattr(args, "num_videos_per_prompt", 1))
    
    if vbench_json_path:
        if not os.path.isfile(vbench_json_path):
            raise ValueError(f"JSON file not found: {vbench_json_path}")
        with open(vbench_json_path, 'r') as f:
            prompts_data = json.load(f)
        if index_end < 0 or index_end >= len(prompts_data):
            index_end = len(prompts_data) - 1
        selected_prompts = prompts_data[index_start:index_end+1]
    else:
        # Original logic: Single prompt only
        selected_prompts = [{"prompt_en": args.prompt}]
    
    # Load model
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` does not exist: {models_root_path}")
    
    # Create save directory
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)
    
    # Load the sampler (only load the model once)
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # Update sampler internal parameters
    args = hunyuan_video_sampler.args
    model_kwargs = {
        'mode': args.mode, 
        'max_order': args.max_order,
        'fresh_threshold': args.fresh_threshold, 
        'propagation_ratio': args.propagation_ratio,
        'cluster_num': args.cluster_num,
        'k': args.k,
        'T-wise': args.T_wise,
    }
    total_prompts = len(selected_prompts)
    for idx, item in enumerate(selected_prompts):
        prompt_text = item.get("prompt_en", "")
        logger.info(f"Starting inference for Prompt [{idx+1}/{total_prompts}]: {prompt_text}")
        for seed_offset in range(num_videos_per_prompt):
            current_seed = args.seed + seed_offset
            cur_save_path = f"{save_path}/{prompt_text}-{seed_offset}.mp4"

            # Check if the target file already exists
            if os.path.exists(cur_save_path):
                logger.info(f"Video already exists, skipping: {cur_save_path}")
                continue  # Skip this video and proceed to the next one

            with suppress_output():
                outputs = hunyuan_video_sampler.predict(
                    prompt=prompt_text,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=current_seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=1,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                    model_kwargs=model_kwargs        
                )
            samples = outputs['samples']
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f"Sample saved to: {cur_save_path}")

    
if __name__ == "__main__":
    main()
