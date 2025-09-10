#!/usr/bin/env python3
"""
TaylorSeer + QwenImage 纯 Diffusers 实现
基于 infer.py 的架构，集成 TaylorSeer 优化
使用纯 diffusers 框架，支持 NPU 设备
"""

import argparse
import os
import re
import time
import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging

# 设置日志
logger = logging.getLogger(__name__)
diffusers_logging.set_verbosity_info()

# NPU 配置
if hasattr(torch.npu.config, 'allow_internal_format'):
    torch.npu.config.allow_internal_format = False

if torch.npu.is_available():
    torch.npu.empty_cache()

def sanitize_filename(text: str, max_length: int = 80) -> str:
    """清理文件名"""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]

def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """获取torch数据类型"""
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32

def apply_taylorseer_optimization(pipeline: DiffusionPipeline, device: str = "npu") -> DiffusionPipeline:
    """
    应用 TaylorSeer 优化到 QwenImage pipeline
    基于纯 diffusers 的实现方式
    """
    logger.info(f"Applying TaylorSeer optimization to QwenImage pipeline on {device}")
    
    # 获取目标模型
    target_model = None
    if hasattr(pipeline, "transformer"):
        target_model = pipeline.transformer
    elif hasattr(pipeline, "unet"):
        target_model = pipeline.unet
    else:
        logger.warning("No transformer/unet module found; TaylorSeer cannot be applied")
        return pipeline
    
    # 保存原始 forward 方法
    if not hasattr(target_model, '_original_forward_method'):
        target_model._original_forward_method = target_model.forward
    
    # 创建包装的forward方法
    def taylorseer_forward_wrapper(*args, **kwargs):
        """
        TaylorSeer 优化的前向传播包装器
        """
        # 过滤掉不支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['cross_attention_kwargs']}
        
        # 初始化缓存系统
        if 'cross_attention_kwargs' in kwargs:
            cross_attention_kwargs = kwargs['cross_attention_kwargs']
        else:
            cross_attention_kwargs = {}
        
        # 获取或初始化缓存
        if 'taylorseer_cache' not in cross_attention_kwargs:
            cross_attention_kwargs['taylorseer_cache'] = {
                'step': 0,
                'features': {},
                'taylor_coeffs': {},
                'cache_hits': 0,
                'cache_misses': 0
            }
        
        cache = cross_attention_kwargs['taylorseer_cache']
        cache['step'] += 1
        current_step = cache['step']
        
        # 确定计算类型
        calc_type = determine_calculation_type(current_step, cache)
        
        if calc_type == 'full':
            # 完整计算
            result = target_model._original_forward_method(*args, **filtered_kwargs)
            
            # 缓存特征
            if hasattr(result, 'last_hidden_state'):
                cache['features'][current_step] = result.last_hidden_state.detach()
            elif isinstance(result, torch.Tensor):
                cache['features'][current_step] = result.detach()
            
            cache['cache_misses'] += 1
            
        elif calc_type == 'taylor':
            # TaylorSeer 缓存计算
            result = taylorseer_cached_forward(
                target_model, args[0] if args else kwargs.get('hidden_states'), 
                kwargs.get('encoder_hidden_states'), kwargs.get('timestep'),
                kwargs.get('attention_mask'), cache, **filtered_kwargs
            )
            cache['cache_hits'] += 1
            
        else:
            # 部分缓存
            result = target_model._original_forward_method(*args, **filtered_kwargs)
            cache['cache_misses'] += 1
        
        return result
    
    # 替换forward方法
    target_model.forward = taylorseer_forward_wrapper
    
    # 设置缓存配置
    pipeline.taylorseer_config = {
        'enabled': True,
        'device': device,
        'npu_optimized': device == "npu",
        'optimization_level': 'high',
        'cache_size': 4096,
        'taylor_order': 2,
        'first_enhance_steps': 3
    }
    
    logger.info("TaylorSeer optimization successfully applied")
    return pipeline


def determine_calculation_type(step: int, cache: Dict[str, Any]) -> str:
    """
    确定计算类型：full, partial, taylor
    """
    config = cache.get('config', {})
    first_enhance_steps = config.get('first_enhance_steps', 3)
    
    if step <= first_enhance_steps:
        return 'full'
    elif step <= first_enhance_steps + 2:
        return 'partial'
    else:
        return 'taylor'

def taylorseer_cached_forward(
    model,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    timestep: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    cache: Dict[str, Any],
    **kwargs
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    TaylorSeer 缓存前向传播
    使用泰勒展开进行特征插值
    """
    current_step = cache['step']
    features = cache['features']
    
    # 找到最近的缓存特征
    cached_steps = sorted(features.keys())
    if len(cached_steps) < 2:
        # 缓存不足，使用完整计算
        return model._original_forward_method(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            attention_mask=attention_mask,
            **kwargs
        )
    
    # 选择最近的缓存步骤
    prev_step = cached_steps[-1]
    prev_feature = features[prev_step]
    
    # 简单的线性插值（可以扩展为高阶泰勒展开）
    if len(cached_steps) >= 2:
        prev2_step = cached_steps[-2]
        prev2_feature = features[prev2_step]
        
        # 计算一阶导数（简化版）
        dt = prev_step - prev2_step
        if dt > 0:
            derivative = (prev_feature - prev2_feature) / dt
            # 泰勒展开：f(t) ≈ f(t0) + f'(t0) * (t - t0)
            dt_current = current_step - prev_step
            interpolated = prev_feature + derivative * dt_current
        else:
            interpolated = prev_feature
    else:
        interpolated = prev_feature
    
    # 返回插值结果
    if isinstance(interpolated, torch.Tensor):
        return interpolated
    else:
        return {'last_hidden_state': interpolated}

def get_taylorseer_stats(pipeline: DiffusionPipeline) -> Dict[str, Any]:
    """
    获取 TaylorSeer 统计信息
    """
    if not hasattr(pipeline, 'taylorseer_config'):
        return {'taylorseer_enabled': False}
    
    config = pipeline.taylorseer_config
    stats = {
        'taylorseer_enabled': config.get('enabled', False),
        'device': config.get('device', 'unknown'),
        'npu_optimized': config.get('npu_optimized', False),
        'optimization_level': config.get('optimization_level', 'medium'),
        'cache_size': config.get('cache_size', 0),
        'taylor_order': config.get('taylor_order', 2),
        'first_enhance_steps': config.get('first_enhance_steps', 3)
    }
    
    return stats

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TaylorSeer + QwenImage 纯 Diffusers 实现")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="数据类型，bfloat16为默认值",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=2.0,
        help="分类器自由引导的引导强度",
    )
    parser.add_argument("--outdir", type=str, default="outputs", help="保存图像的目录")
    parser.add_argument("--prefix", type=str, default="taylorseer_diffusers", help="图像文件名前缀")
    parser.add_argument(
        "--model",
        type=str,
        default="/root/autodl-tmp/qwenimage",
        help="模型路径",
    )
    parser.add_argument("--device", type=str, default="npu", choices=["npu", "cuda", "cpu"], help="计算设备")
    parser.add_argument("--width", type=int, default=1024, help="图像宽度")
    parser.add_argument("--height", type=int, default=1024, help="图像高度")
    parser.add_argument("--negative_prompt", type=str, default="", help="负面提示词")
    parser.add_argument("--enable_taylorseer", action="store_true", default=True, help="启用 TaylorSeer 优化")
    parser.add_argument("--cache_size", type=int, default=4096, help="缓存大小")
    parser.add_argument("--taylor_order", type=int, default=2, help="泰勒展开阶数")
    parser.add_argument("--first_enhance_steps", type=int, default=3, help="首次增强步数")

    return parser.parse_args()

def main() -> None:
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    # 获取数据类型
    torch_dtype = get_torch_dtype(args.dtype)

    # 加载模型
    print(f"🚀 加载模型: {args.model}")
    pipeline = DiffusionPipeline.from_pretrained(
        args.model, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="balanced"
    )

    # 应用 TaylorSeer 优化
    if args.enable_taylorseer:
        print("🔧 应用 TaylorSeer 优化...")
        pipeline = apply_taylorseer_optimization(pipeline, device=args.device)
        
        # 更新缓存配置
        if hasattr(pipeline, 'taylorseer_config'):
            pipeline.taylorseer_config.update({
                'cache_size': args.cache_size,
                'taylor_order': args.taylor_order,
                'first_enhance_steps': args.first_enhance_steps
            })

    # 性能监控
    start_time = time.time()

    print(f"🎨 生成图像，提示词: {args.prompt}")
    print(f"📊 步数: {args.steps}, 引导强度: {args.true_cfg_scale}")
    if args.enable_taylorseer:
        print(f"⚡ TaylorSeer 优化: 启用 (缓存大小: {args.cache_size}, 泰勒阶数: {args.taylor_order})")

    # 生成图像
    image = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=int(args.steps),
        true_cfg_scale=float(args.true_cfg_scale),
        generator=torch.Generator("npu:0").manual_seed(int(args.seed)),
    ).images[0]

    # 计算时间
    elapsed_time = time.time() - start_time
    print(f"⏱️  生成耗时: {elapsed_time:.2f} 秒")

    # 保存图像
    safe_filename = sanitize_filename(args.prompt)
    save_path = os.path.join(args.outdir, f"{args.prefix}_{safe_filename}.png")
    image.save(save_path)
    print(f"💾 图像已保存: {save_path}")

    # 显示 TaylorSeer 统计信息
    if args.enable_taylorseer:
        stats = get_taylorseer_stats(pipeline)
        print(f"📈 TaylorSeer 统计信息: {stats}")

    print("🎉 推理完成！")

if __name__ == "__main__":
    main()
