import argparse
import os
import re
import time
from typing import List, Optional

import torch
import torch_npu 
from diffusers import DiffusionPipeline

from forwards import (
    clusca_flux_single_block_forward,
    clusca_flux_double_block_forward,
    clusca_flux_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-NPU batch inference for FLUX with ClusCa overrides",
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
        default="black-forest-labs/FLUX.1-dev",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps (num_inference_steps).",
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
        default="npu",
        choices=["npu", "cpu"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
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
        default="ClusCa",
        help="Filename prefix for saved images.",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable CPU offload to reduce VRAM usage.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Optional image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Optional image width.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Optional guidance scale (if supported by the pipeline).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional maximum number of prompts to process.",
    )

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

    # ClusCa settings and forward overrides
    pipeline.transformer.__class__.num_steps = int(args.steps)
    pipeline.transformer.__class__.forward = clusca_flux_forward
    for double_transformer_block in pipeline.transformer.transformer_blocks:
        double_transformer_block.__class__.forward = clusca_flux_double_block_forward
    for single_transformer_block in pipeline.transformer.single_transformer_blocks:
        single_transformer_block.__class__.forward = clusca_flux_single_block_forward

    if args.enable_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(args.device)

    prompts = read_prompts(args.prompt_file, args.max_images)
    if len(prompts) == 0:
        print("No prompts found. Exiting.")
        return

    total_time_s = 0.0
    num_images = 0

    is_npu = args.device == "npu" and hasattr(torch, 'npu') and torch.npu.is_available()
    if is_npu:
        parameter_peak_memory = torch.npu.max_memory_allocated(device="npu")
        torch.npu.reset_peak_memory_stats()

    for index, prompt in enumerate(prompts):
        effective_seed = args.seed + index if args.unique_seed_per_prompt else args.seed
        generator = torch.Generator(device="cpu").manual_seed(effective_seed)

        height_kw = {"height": args.height} if args.height is not None else {}
        width_kw = {"width": args.width} if args.width is not None else {}
        guidance_kw = (
            {"guidance_scale": float(args.guidance_scale)} if args.guidance_scale is not None else {}
        )

        if is_npu:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        image = pipeline(
            prompt,
            num_inference_steps=int(args.steps),
            generator=generator,
            **height_kw,
            **width_kw,
            **guidance_kw,
        ).images[0]

        if is_npu:
            end.record()
            torch.npu.synchronize()
            elapsed_time_s = start.elapsed_time(end) * 1e-3
        else:
            elapsed_time_s = time.time() - start_time

        total_time_s += elapsed_time_s
        num_images += 1

        safe = sanitize_filename(prompt)
        filename = f"{args.prefix}_{index:04d}_{safe}.png"
        save_path = os.path.join(args.outdir, filename)
        image.save(save_path)
        print(f"Saved: {save_path} | time: {elapsed_time_s:.2f}s")

    if is_npu:
        peak_memory = torch.npu.max_memory_allocated(device="npu")
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


