import argparse
import os
import time
import logging
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from .forwards.qwen_forward_optimized import apply_taylorseer_to_qwen_optimized, get_taylorseer_stats_optimized
from .cache_functions import get_cache_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():

    parser = argparse.ArgumentParser(description="QwenImage + TaylorSeer 单次推理")
    

    parser.add_argument("--model", type=str, default="Qwen/QwenImage-1.5", 
                       help="QwenImage模型路径或HuggingFace ID")
    parser.add_argument("--device", type=str, default="npu", choices=["npu", "cpu", "cuda"],
                       help="计算设备")
    

    parser.add_argument("--prompt", type=str, required=True, help="文本提示词")
    parser.add_argument("--negative_prompt", type=str, default="", help="负面提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="引导强度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    

    parser.add_argument("--height", type=int, default=1024, help="图像高度")
    parser.add_argument("--width", type=int, default=1024, help="图像宽度")
    parser.add_argument("--num_images", type=int, default=1, help="生成图像数量")

    parser.add_argument("--dtype", type=str, default="fp16", 
                       choices=["fp16", "bf16", "fp32"], help="计算精度")
    

    parser.add_argument("--outdir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--prefix", type=str, default="qwen_taylorseer", help="文件名前缀")
    

    parser.add_argument("--enable_taylorseer", action="store_true", 
                       help="启用TaylorSeer缓存机制")
    parser.add_argument("--cache_level", type=str, default="high", 
                       choices=["low", "medium", "high"], help="缓存优化级别")
    

    parser.add_argument("--npu_optimization", action="store_true", 
                       help="启用NPU特定优化")
    parser.add_argument("--memory_efficient", action="store_true", 
                       help="启用内存效率优化")
    
    return parser.parse_args()

def setup_device(device: str, dtype: str):

    if device == "npu":
        if not hasattr(torch, 'npu'):
            logger.warning("NPU not available, falling back to CPU")
            device = "cpu"
        else:
            logger.info("Using NPU device")
            torch.npu.set_device(0)  # 使用第一个NPU设备
    elif device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            logger.info("Using CUDA device")

    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    return device, torch_dtype

def load_model(model_path: str, torch_dtype: torch.dtype, device: str):

    logger.info(f"Loading model: {model_path}")
    
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True
        )

        if device == "npu":
            pipeline = pipeline.to("npu")
        else:
            pipeline = pipeline.to(device)
        
        logger.info("Model loaded successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def apply_memory_efficient_options(pipeline, args)
    if not args.memory_efficient:
        return pipeline

    try:
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing("max")
            logger.info("Enabled attention slicing")
    except Exception as e:
        logger.debug(f"enable_attention_slicing not applied: {e}")

    try:
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
    except Exception as e:
        logger.debug(f"enable_vae_slicing not applied: {e}")

    try:
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
        elif hasattr(getattr(pipeline, "vae", None), "enable_tiling"):
            pipeline.vae.enable_tiling()
            logger.info("Enabled VAE tiling via VAE module")
    except Exception as e:
        logger.debug(f"VAE tiling not applied: {e}")

    return pipeline

def apply_taylorseer_optimizations(pipeline, args):
    if not args.enable_taylorseer:
        logger.info("TaylorSeer disabled, using standard pipeline")
        return pipeline
    
    logger.info("Applying TaylorSeer optimizations...")
    
    try:
        pipeline = apply_taylorseer_to_qwen_optimized(pipeline, device=args.device)
        cache_config = get_cache_config()
        cache_config['cache_optimization_level'] = args.cache_level
        cache_config['npu_optimized'] = args.device == "npu"
        cache_config['memory_efficient'] = args.memory_efficient

        if hasattr(pipeline, 'cache_config'):
            pipeline.cache_config.update(cache_config)
        
        logger.info("TaylorSeer optimizations applied successfully")
        
    except Exception as e:
        logger.warning(f"Failed to apply TaylorSeer optimizations: {e}")
        logger.info("Continuing with standard pipeline")
    
    return pipeline

def generate_image(pipeline, args):

    logger.info("Starting image generation...")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed)
        elif args.device == "npu":
            torch.npu.manual_seed(args.seed)
    
    generation_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "num_images_per_prompt": args.num_images,
    }
    
    core_module = getattr(pipeline, 'transformer', None) or getattr(pipeline, 'unet', None)
    taylorseer_ctx = None
    if args.enable_taylorseer and core_module is not None:
        taylorseer_ctx = {
            'cache_dic': {},
            'current': {}
        }
        try:
            setattr(core_module, '_taylorseer_kwargs', taylorseer_ctx)
        except Exception:
            pass
    start_time = time.time()
    
    try:
        with torch.no_grad():
            result = pipeline(**generation_kwargs)
        
        generation_time = time.time() - start_time
        
        logger.info(f"Image generation completed in {generation_time:.2f} seconds")
        
        return result, generation_time
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise
    finally:
        if taylorseer_ctx is not None and core_module is not None:
            try:
                delattr(core_module, '_taylorseer_kwargs')
            except Exception:
                pass

def save_images(result, args, generation_time):
    os.makedirs(args.outdir, exist_ok=True)
    

    if isinstance(result, dict) and 'images' in result:
        images = result['images']
    elif isinstance(result, list):
        images = result
    else:
        images = [result]
    
    saved_paths = []
    
    for i, image in enumerate(images):
        timestamp = int(time.time())
        filename = f"{args.prefix}_{timestamp}_{i:02d}.png"
        filepath = os.path.join(args.outdir, filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            if hasattr(image, 'cpu'):
                image = image.cpu()
            if hasattr(image, 'numpy'):
                image = image.numpy()
            pil_image = Image.fromarray(image)
            pil_image.save(filepath)
        
        saved_paths.append(filepath)
        logger.info(f"Saved image: {filepath}")

    info_file = os.path.join(args.outdir, f"{args.prefix}_{timestamp}_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Negative Prompt: {args.negative_prompt}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Guidance Scale: {args.guidance_scale}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Size: {args.width}x{args.height}\n")
        f.write(f"Generation Time: {generation_time:.2f}s\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"TaylorSeer: {args.enable_taylorseer}\n")
        f.write(f"Cache Level: {args.cache_level}\n")
        f.write(f"Saved Images: {', '.join(saved_paths)}\n")
    
    logger.info(f"Generation info saved: {info_file}")
    return saved_paths

def main():
    args = parse_args()
    
    logger.info("Starting QwenImage + TaylorSeer inference")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # 提高数值稳定性：关闭 NPU 内部格式
        if hasattr(torch, 'npu') and hasattr(torch.npu, 'config') and hasattr(torch.npu.config, 'allow_internal_format'):
            torch.npu.config.allow_internal_format = False
        # 设置设备
        device, torch_dtype = setup_device(args.device, args.dtype)
        args.device = device  # 更新设备参数
        
        # 加载模型
        pipeline = load_model(args.model, torch_dtype, device)
        
        # 应用TaylorSeer优化
        pipeline = apply_taylorseer_optimizations(pipeline, args)

        # 应用内存优化选项（attention/vae slicing 与 tiling）
        pipeline = apply_memory_efficient_options(pipeline, args)
        
        # 生成图像
        result, generation_time = generate_image(pipeline, args)
        
        # 保存图像
        saved_paths = save_images(result, args, generation_time)
        
        # 显示TaylorSeer统计信息
        if args.enable_taylorseer:
            stats = get_taylorseer_stats_optimized(pipeline)
            logger.info(f"TaylorSeer Stats: {stats}")
        
        # 显示性能指标
        logger.info(f"Performance Summary:")
        logger.info(f"  - Generation Time: {generation_time:.2f}s")
        logger.info(f"  - Images Generated: {len(saved_paths)}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Precision: {args.dtype}")
        logger.info(f"  - TaylorSeer: {'Enabled' if args.enable_taylorseer else 'Disabled'}")
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()

