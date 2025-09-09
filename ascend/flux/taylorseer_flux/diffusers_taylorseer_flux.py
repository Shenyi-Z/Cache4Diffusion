import argparse
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch_npu
from diffusers import DiffusionPipeline
from diffusers.utils import logging

from forwards import (
    taylorseer_flux_single_block_forward,
    taylorseer_flux_double_block_forward,
    taylorseer_flux_forward,
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
    parser = argparse.ArgumentParser(description="Single-prompt inference for FLUX with TaylorSeer overrides")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate an image for.")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps (num_inference_steps).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable CPU offload to reduce VRAM usage.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (if supported).",
    )
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save the image.")
    parser.add_argument("--prefix", type=str, default="TaylorSeer", help="Filename prefix for the image.")
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--device", type=str, default="npu", choices=["npu", "cpu"], help="Compute device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("Requested low-precision dtype on CPU; overriding to float32 for compatibility.")
        torch_dtype = torch.float32

    pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)

    # TaylorSeer settings and forward overrides
    pipeline.transformer.__class__.num_steps = int(args.steps)
    pipeline.transformer.__class__.forward = taylorseer_flux_forward
    for double_transformer_block in pipeline.transformer.transformer_blocks:
        double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
    for single_transformer_block in pipeline.transformer.single_transformer_blocks:
        single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

    if args.enable_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(args.device)

    is_npu = args.device == "npu" and hasattr(torch, 'npu') and torch.npu.is_available()
    if is_npu:
        parameter_peak_memory = torch.npu.max_memory_allocated(device="npu")
        torch.npu.reset_peak_memory_stats()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()

    image = pipeline(
        args.prompt,
        num_inference_steps=int(args.steps),
        generator=torch.Generator("cpu").manual_seed(int(args.seed)),
        guidance_scale=float(args.guidance_scale),
    ).images[0]

    if is_npu:
        end.record()
        torch.npu.synchronize() 
        elapsed_time = start.elapsed_time(end) * 1e-3
        peak_memory = torch.npu.max_memory_allocated(device="npu")
        print(
          f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    else:
        elapsed_time = time.time() - start_time
        print(f"elapsed time: {elapsed_time:.2f} sec")

    safe = sanitize_filename(args.prompt)
    save_path = os.path.join(args.outdir, f"{args.prefix}_{safe}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()