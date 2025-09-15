import argparse
import os
import re
import time
from typing import List, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from forwards import (
    taylorseer_hunyuan_video_single_block_forward,
    taylorseer_hunyuan_video_double_block_forward,
    taylorseer_hunyuan_video_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU batch inference for HunyuanVideo with TaylorSeer overrides",
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuanvideo-community/HunyuanVideo",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--video-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(720, 1280),
                        help="Video size as two integers: HEIGHT WIDTH (e.g. --video-size 720 1280).")
    parser.add_argument("--video-length", type=int, default=129, help="Number of frames in the generated video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument(
        "--infer-steps",
        type=int,
        default=50,
        help="Number of sampling steps (num_inference_steps).",
    )
    parser.add_argument(
        "--embedded-cfg-scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--unique_seed_per_prompt",
        action="store_true",
        help="If set, increment seed by line index for each prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="TaylorSeer",
        help="Filename prefix for saved images.",
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Enable CPU offload to reduce VRAM usage.",
    )
    parser.add_argument("--use_taylor", action="store_true", help="Use TaylorSeer speedup.")
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional maximum number of prompts to process.",
    )
    parser.add_argument("--save-path", type=str, default="outputs", help="Directory to save the image.")

    return parser.parse_args()


def read_prompts(prompt_file: str, max_images: Optional[int] = None) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    prompts = [line for line in lines if len(line) > 0]
    if max_images is not None:
        prompts = prompts[:max_images]
    return prompts


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


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("Requested low-precision dtype on CPU; overriding to float32 for compatibility.")
        torch_dtype = torch.float32

    print(f"Loading pipeline: {args.model} (dtype={torch_dtype}, device={args.device})")
    pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)

    pipeline.vae.enable_tiling()
    pipeline.to(args.device)

    # TaylorSeer settings and forward overrides
    if args.use_taylor:
        pipeline.transformer.__class__.num_steps = int(args.infer_steps)
        pipeline.transformer.__class__.forward = taylorseer_hunyuan_video_forward
        for double_transformer_block in pipeline.transformer.transformer_blocks:
            double_transformer_block.__class__.forward = taylorseer_hunyuan_video_double_block_forward
        for single_transformer_block in pipeline.transformer.single_transformer_blocks:
            single_transformer_block.__class__.forward = taylorseer_hunyuan_video_single_block_forward

    if args.use_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(args.device)

    prompts = read_prompts(args.prompt_file, args.max_videos)
    if len(prompts) == 0:
        print("No prompts found. Exiting.")
        return

    total_time_s = 0.0
    num_images = 0

    is_cuda = args.device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()

    height, width = args.video_size
    for index, prompt in enumerate(prompts):
        effective_seed = args.seed + index if args.unique_seed_per_prompt else args.seed
        generator = torch.Generator(device="cpu").manual_seed(effective_seed)

        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=args.video_length,
            num_inference_steps=args.infer_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            guidance_scale=args.embedded_cfg_scale,
            attention_kwargs={},  # pass attention_kwargs as not None to ensure TaylorSeer works properly,
        ).frames[0]

        if is_cuda:
            end.record()
            torch.cuda.synchronize()
            elapsed_time_s = start.elapsed_time(end) * 1e-3
        else:
            elapsed_time_s = time.time() - start_time

        total_time_s += elapsed_time_s
        num_images += 1

        save = sanitize_filename(prompt)
        save_path = os.path.join(args.save_path, f"{args.prefix}_{save}.mp4")
        export_to_video(output, save_path, fps=args.fps)
        print(f"Saved: {save_path} | time: {elapsed_time_s:.2f}s")

    if is_cuda:
        peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        print(
            f"Processed {num_images} images | avg time: {total_time_s / max(num_images, 1):.2f}s | "
            f"parameter memory: {parameter_peak_memory/1e9:.2f} GB | peak memory: {peak_memory/1e9:.2f} GB"
        )
    else:
        print(
            f"Processed {num_images} images | avg time: {total_time_s / max(num_images, 1):.2f}s"
        )


if __name__ == "__main__":
    main()


