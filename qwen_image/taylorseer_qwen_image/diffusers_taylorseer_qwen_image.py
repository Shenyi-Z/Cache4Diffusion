import argparse
import os
import re
import time

import torch
from diffusers import DiffusionPipeline, QwenImagePipeline
from diffusers.utils import logging

from forwards import (
    taylorseer_qwen_image_mmdit_forward,
    taylorseer_qwen_image_forward,
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
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-prompt inference for Qwen-Image with TaylorSeer overrides")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate an image for.")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps (num_inference_steps).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights. bfloat16 for default",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable CPU offload to reduce VRAM usage.",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (if supported).",
    )
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save the image.")
    parser.add_argument("--prefix", type=str, default="TaylorSeer", help="Filename prefix for the image.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-Image",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device.")
    parser.add_argument(
        "--use_taylor",
        action="store_true",
        help="Use TaylorSeer overrides for the transformer blocks.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.bfloat16, torch.float16):
        print("Requested low-precision dtype on CPU; overriding to float32 for compatibility.")
        torch_dtype = torch.float32
    
    pipeline = DiffusionPipeline.from_pretrained(
        args.model, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map='cuda'
    )

    if args.use_taylor:
        # TaylorSeer settings and forward overrides
        pipeline.transformer.__class__.num_steps = int(args.steps)
        # pipeline.transformer.__class__.forward = taylorseer_qwen_image_forward
        pipeline.transformer.forward = taylorseer_qwen_image_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
        for transformer_block in pipeline.transformer.transformer_blocks:
            transformer_block.forward = taylorseer_qwen_image_mmdit_forward.__get__(transformer_block, transformer_block.__class__)
        # pipeline.__class__.__call__ = taylorseer_qwen_image_pipeline_call  # OOM if use this replaced call method. No parallelism, only set device_map='balanced' when load pipe

    if args.enable_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")
        pipeline.enable_model_cpu_offload()
    else:
        # pipeline.to(args.device)
        ...

    is_cuda = args.device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()

    image = pipeline(
        prompt=args.prompt,
        # negative_prompt="",
        # width=int(args.width) if hasattr(args, 'width') else 1024,
        # height=int(args.height) if hasattr(args, 'height') else 1024,
        num_inference_steps=int(args.steps),
        true_cfg_scale=float(args.true_cfg_scale),
        generator=torch.Generator("cuda").manual_seed(int(args.seed)),
    ).images[0]

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

    safe = sanitize_filename(args.prompt)
    save_path = os.path.join(args.outdir, f"{args.prefix}_{safe}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()