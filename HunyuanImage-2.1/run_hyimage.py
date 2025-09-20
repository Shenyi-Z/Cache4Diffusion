import torch
import argparse
import time
import re
import os
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
import loguru

def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Clean and sanitize text for use as filename."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using HunyuanImage model')
    
    parser.add_argument('--prompt', type=str, 
                       default="A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word \"Tencent\" on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style.",
                       help='Text prompt for image generation')
    
    parser.add_argument('--model_name', type=str, 
                       default="hunyuanimage-v2.1",
                       choices=["hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"],
                       help='Model name to use (default: hunyuanimage-v2.1)')
    
    parser.add_argument('--use_reprompt', action='store_true', default=True,
                       help='Enable prompt enhancement (default: True)')

    parser.add_argument('--use_refiner', action='store_true', default=True,
                       help='Enable refiner model (default: True)')
    
    parser.add_argument('--shift', type=float, default=5.0,
                       help='Shift parameter (default: 5.0)')
    
    parser.add_argument('--seed', type=int, default=649151,
                       help='Random seed for generation (default: 649151)')
    
    parser.add_argument('--width', type=int, default=2048,
                       help='Image width (default: 2048)')
    
    parser.add_argument('--height', type=int, default=2048,
                       help='Image height (default: 2048)')
    
    parser.add_argument('--guidance_scale', type=float, default=3.5,
                       help='Guidance scale (default: 3.5)')
    
    parser.add_argument('--outdir', type=str, default="outputs",
                       help='Directory to save the image (default: outputs)')
    
    parser.add_argument('--prefix', type=str, default="",
                       help='Filename prefix for the image (default: no prefix)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    loguru.logger.info(f"Loading model: {args.model_name}")
    pipe = HunyuanImagePipeline.from_pretrained(model_name=args.model_name, torch_dtype='bf16')
    pipe = pipe.to("cuda")
    
    # Memory tracking setup
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()
    
    loguru.logger.info(f"Generating image with prompt: {args.prompt}")
    loguru.logger.info(f"Parameters: reprompt={args.use_reprompt}, refiner={args.use_refiner}, shift={args.shift}, seed={args.seed}")
    # Determine number of inference steps based on model type
    num_inference_steps = 8 if "distilled" in args.model_name else 50
    
    image = pipe(
        prompt=args.prompt,
        # Examples of supported resolutions and aspect ratios for HunyuanImage-2.1:
        # 16:9  -> width=2560, height=1536
        # 4:3   -> width=2304, height=1792
        # 1:1   -> width=2048, height=2048
        # 3:4   -> width=1792, height=2304
        # 9:16  -> width=1536, height=2560
        # Please use one of the above width/height pairs for best results.
        width=args.width,
        height=args.height,
        use_reprompt=args.use_reprompt,
        use_refiner=args.use_refiner,
        num_inference_steps=num_inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed,
    )
    
    # Performance measurement and reporting
    if is_cuda:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) * 1e-3
        peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        loguru.logger.info(
            f"Performance stats - Time: {elapsed_time:.2f}s, Parameter memory: {parameter_peak_memory/1e9:.2f}GB, Peak memory: {peak_memory/1e9:.2f}GB"
        )
    else:
        elapsed_time = time.time() - start_time
        loguru.logger.info(f"Performance stats - Time: {elapsed_time:.2f}s")
    
    # Generate filename from prompt
    safe_prompt = sanitize_filename(args.prompt, max_length=100)
    
    # Build filename with optional prefix
    if args.prefix:
        filename = f"{args.prefix}_{safe_prompt}.png"
    else:
        filename = f"{safe_prompt}.png"
    
    output_path = os.path.join(args.outdir, filename)
    
    loguru.logger.info(f"Saving image to: {output_path}")
    image.save(output_path)
    loguru.logger.info("Image generation completed!")

if __name__ == "__main__":
    main()

