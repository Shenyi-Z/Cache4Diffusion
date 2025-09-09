"""
Image generation script for GenEval using Qwen-Image,
with support for TaylorSeer optimizations.
Adapted from diffusers_generate.py(geneval) and batch_infer.py.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from pytorch_lightning import seed_everything
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

# --- Functions from batch_infer.py (or similar logic) ---
# If TaylorSeer is used, you need to import its forward functions.
# Make sure the 'forwards' module is accessible from this script's location.
# For example, by adding its path to sys.path or placing it correctly.
from forwards import taylorseer_qwen_image_forward, taylorseer_qwen_image_mmdit_forward

def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images for GenEval using Qwen-Image.")
    parser.add_argument(
        "--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen-Image", help="Huggingface model name, should be Qwen-Image."
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs", help="Directory to write results to."
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of samples per prompt."
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of DDIM sampling steps."
    )
    parser.add_argument(
        "--H", type=int, default=1024, help="Image height in pixels."
    )
    parser.add_argument(
        "--W", type=int, default=1024, help="Image width in pixels."
    )
    parser.add_argument(
        "--scale", type=float, default=7.5, help="Guidance scale."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The seed for reproducible sampling."
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Torch dtype."
    )
    # TaylorSeer specific argument
    parser.add_argument(
        "--use_taylor", action="store_true", help="If set, enables TaylorSeer optimizations."
    )
    # TODO: dpm++ sampler seems to be conflict with QwenImagePipeline, whose timesteps use custom 'sigmas'.
    # Customize __call__ method of pipeline may resolve this. But that may cause OOM.
    parser.add_argument(
        "--sampler", type=str, default="default", help="Sampler to use. E.g., 'dpm++'. Default is the pipeline's default."
    )
    opt = parser.parse_args()
    return opt

def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Load model (logic from batch_infer.py)
    torch_dtype = get_torch_dtype(opt.dtype)
    print(f"Loading pipeline: {opt.model} (dtype={torch_dtype})")
    
    # Use device_map for multi-GPU setup like in batch_infer.py
    pipeline = DiffusionPipeline.from_pretrained(
        opt.model, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="balanced" # Or "auto" or a specific device
    )

    if opt.sampler.lower() == 'dpm++':
        print("Switching to DPM++ (DPMSolverMultistepScheduler) sampler.")
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    if opt.use_taylor:
        print("Applying TaylorSeer optimizations.")
        # This part requires the forward functions to be available.
        # You might need to adjust imports based on your project structure.
        try:
            pipeline.transformer.__class__.num_steps = int(opt.steps)
            pipeline.transformer.forward = taylorseer_qwen_image_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
            for transformer_block in pipeline.transformer.transformer_blocks:
                transformer_block.forward = taylorseer_qwen_image_mmdit_forward.__get__(transformer_block, transformer_block.__class__)
        except ImportError:
            print("Warning: 'forwards' module not found. TaylorSeer optimizations will not be applied.")


    # Loop through prompts and generate images
    for index, metadata in enumerate(tqdm(metadatas, desc="Processing prompts")):
        # Use a unique seed for each prompt if desired, or a fixed one
        seed_everything(opt.seed + index)
        generator = torch.Generator(device="cpu").manual_seed(opt.seed + index)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        print(f"\nPrompt ({index + 1}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        # Generate n_samples for the prompt
        for i in range(opt.n_samples):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=opt.steps,
                guidance_scale=opt.scale,
                height=opt.H,
                width=opt.W,
                generator=generator,
            ).images[0]
            
            image.save(os.path.join(sample_path, f"{i:05}.png"))

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)