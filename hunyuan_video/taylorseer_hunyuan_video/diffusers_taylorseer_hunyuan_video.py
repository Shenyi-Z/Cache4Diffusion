import argparse
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import logging, export_to_video

from forwards import (
    taylorseer_hunyuan_video_single_block_forward,
    taylorseer_hunyuan_video_double_block_forward,
    taylorseer_hunyuan_video_forward,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def sanitize_filename(text: str, max_length: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-prompt inference for HunyuanVideo with TaylorSeer overrides")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate an image for.")
    parser.add_argument("--video-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(720, 1280),
                        help="Video size as two integers: HEIGHT WIDTH (e.g. --video-size 720 1280).")
    parser.add_argument("--video-length", type=int, default=129, help="Number of frames in the generated video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--infer-steps", type=int, default=50, help="Number of sampling steps (num_inference_steps).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Enable CPU offload to reduce VRAM usage.",
    )
    parser.add_argument(
        "--embedded-cfg-scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument("--save-path", type=str, default="outputs", help="Directory to save the image.")
    parser.add_argument("--prefix", type=str, default="TaylorSeer", help="Filename prefix for the image.")
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuanvideo-community/HunyuanVideo",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--use_taylor", action="store_true", help="Use TaylorSeer speedup.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("Requested low-precision dtype on CPU; overriding to float32 for compatibility.")
        torch_dtype = torch.float32

    # 1. load model
    pipeline = DiffusionPipeline.from_pretrained(
        args.model, 
        torch_dtype=torch.float16,
        # device_map='balanced'
    )

    pipeline.vae.enable_tiling()
    pipeline.to("cuda")

    # 2. TaylorSeer settings and forward overrides
    if args.use_taylor:
        print("Using TaylorSeer speedup for HunyuanVideo...")
        pipeline.transformer.__class__.num_steps = int(args.infer_steps)
        # pipeline.transformer.__class__.forward = taylorseer_hunyuan_video_forward
        pipeline.transformer.__class__.forward = taylorseer_hunyuan_video_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
        for double_transformer_block in pipeline.transformer.transformer_blocks:
            # double_transformer_block.__class__.forward = taylorseer_hunyuan_video_double_block_forward
            double_transformer_block.__class__.forward = taylorseer_hunyuan_video_double_block_forward.__get__(double_transformer_block, double_transformer_block.__class__)
        for single_transformer_block in pipeline.transformer.single_transformer_blocks:
            # single_transformer_block.__class__.forward = taylorseer_hunyuan_video_single_block_forward
            single_transformer_block.__class__.forward = taylorseer_hunyuan_video_single_block_forward.__get__(single_transformer_block, single_transformer_block.__class__)
    else:
        print("Using original HunyuanVideo without TaylorSeer...")

    if args.use_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(args.device)
        ...

    # begin of time recording
    is_cuda = args.device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()

    # generate video
    height, width = args.video_size
    output = pipeline(
        prompt=args.prompt,
        height=height,
        width=width,
        num_frames=args.video_length,
        num_inference_steps=args.infer_steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        guidance_scale=args.embedded_cfg_scale,
        attention_kwargs={},  # pass attention_kwargs as not None to ensure TaylorSeer works properly
    ).frames[0]

    # end of time recording
    if is_cuda:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) * 1e-3
        peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    else:
        elapsed_time = time.time() - start_time
        print(f"elapsed time: {elapsed_time:.2f} sec")

    save = sanitize_filename(args.prompt)
    save_path = os.path.join(args.save_path, f"{args.prefix}_{save}.mp4")
    export_to_video(output, save_path, fps=args.fps)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()