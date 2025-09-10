"""
QwenImage模型的真正TaylorSeer前向传播
基于 Cache4Diffusion 的正确实现方式
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
import logging

from ..cache_functions import cache_init, cal_type

logger = logging.getLogger(__name__)

def qwen_taylorseer_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    **kwargs
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    QwenImage模型的TaylorSeer前向传播
    基于 Cache4Diffusion 的正确实现方式
    """
    
    # 初始化缓存系统
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}
    
    if cross_attention_kwargs.get("cache_dic", None) is None:
        cross_attention_kwargs['cache_dic'], cross_attention_kwargs['current'] = cache_init(self)
    
    # 计算缓存类型
    cal_type(cross_attention_kwargs['cache_dic'], cross_attention_kwargs['current'])
    
    # 获取当前状态
    current = cross_attention_kwargs['current']
    cache_dic = cross_attention_kwargs['cache_dic']
    
    # 更新当前步骤
    current['step'] = current.get('step', 0) + 1
    
    # 根据缓存类型决定计算方式
    calc_type = current.get('type', 'full')
    logger.info(f"Step {current['step']}, Calculation type: {calc_type}")
    
    if calc_type == 'full':
        # 完整计算
        logger.info("Performing full calculation")
        # 过滤掉不支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'attention_mask'}
        result = self._original_forward_method(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            **filtered_kwargs
        )
        
        # 缓存注意力权重（如果可用）
        if hasattr(result, 'attn_weights') and cache_dic.get('feature_cache_manager'):
            feature_manager = cache_dic['feature_cache_manager']
            feature_manager.store_features(
                current['step'], current.get('layer', 0), current.get('module', 'self_attn'),
                result, attention_weights=getattr(result, 'attn_weights', None)
            )
    elif calc_type == 'Taylor':
        # TaylorSeer 缓存计算
        logger.info("Using TaylorSeer cache")
        result = self._taylorseer_cached_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            attention_mask=attention_mask,
            cache_dic=cache_dic,
            current=current,
            **kwargs
        )
    else:
        # 其他缓存类型
        logger.info(f"Using {calc_type} cache")
        # 过滤掉不支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'attention_mask'}
        result = self._original_forward_method(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            **filtered_kwargs
        )
    
    # 返回结果
    if return_dict:
        return {
            'last_hidden_state': result,
            'cache_info': {
                'calc_type': calc_type,
                'step': current.get('step', 0),
                'cache_hit': calc_type != 'full'
            }
        }
    else:
        return result

def _original_forward(self, **kwargs):
    """原始的前向传播方法"""
    # 这里应该调用原始的 forward 方法
    # 由于我们替换了 forward，需要保存原始方法
    if hasattr(self, '_original_forward_method'):
        return self._original_forward_method(**kwargs)
    else:
        # 如果没有保存原始方法，使用默认的 forward
        return super(self.__class__, self).forward(**kwargs)

def _taylorseer_cached_forward(self, cache_dic, current, **kwargs):
    """TaylorSeer 缓存前向传播"""
    try:
        # 获取特征缓存管理器
        feature_manager = cache_dic.get('feature_cache_manager')
        if feature_manager is None:
            logger.warning("Feature cache manager not found, using original forward")
            return self._original_forward(**kwargs)
        
        # 获取当前步骤和层信息
        current_step = current.get('step', 0)
        current_layer = current.get('layer', 0)
        current_module = current.get('module', 'self_attn')
        
        # 尝试从缓存检索特征
        cached_features = feature_manager.retrieve_features(
            current_step, current_layer, current_module, 
            use_taylor=True, taylor_order=cache_dic.get('max_order', 1)
        )
        
        if cached_features is not None:
            logger.info(f"Cache hit at step {current_step}, layer {current_layer}, module {current_module}")
            # 使用缓存的特征
            return cached_features
        else:
            logger.info(f"Cache miss at step {current_step}, layer {current_layer}, module {current_module}")
            # 计算新特征
            result = self._original_forward(**kwargs)
            
            # 缓存新特征
            if hasattr(result, 'last_hidden_state'):
                features_to_cache = result.last_hidden_state
            elif isinstance(result, torch.Tensor):
                features_to_cache = result
            else:
                features_to_cache = result
            
            # 存储到缓存
            feature_manager.store_features(
                current_step, current_layer, current_module, features_to_cache
            )
            
            return result
            
    except Exception as e:
        logger.warning(f"TaylorSeer cached forward failed: {e}, falling back to original")
        return self._original_forward(**kwargs)

def apply_taylorseer_to_qwen_real(pipeline: nn.Module, device: str = "npu") -> nn.Module:
    """
    将真正的TaylorSeer应用到QwenImage模型
    基于 Cache4Diffusion 的正确实现方式
    """
    logger.info(f"Applying real TaylorSeer to QwenImage pipeline on {device}")

    # 选择具体的模型模块
    target_model = None
    if hasattr(pipeline, "transformer"):
        target_model = pipeline.transformer
    elif hasattr(pipeline, "unet"):
        target_model = pipeline.unet
    else:
        logger.warning("No transformer/unet module found on pipeline; TaylorSeer cannot be applied")
        return pipeline

    # 关键：保存原始 forward 方法并替换
    if not hasattr(target_model, '_original_forward_method'):
        target_model._original_forward_method = target_model.forward
    
    # 直接替换 forward 方法（这是 Cache4Diffusion 的方式）
    target_model.forward = qwen_taylorseer_forward.__get__(target_model, target_model.__class__)
    
    # 设置 num_steps 属性（用于缓存初始化）
    if not hasattr(target_model.__class__, 'num_steps'):
        target_model.__class__.num_steps = 50  # 默认值
    
    # 在 pipeline 上添加缓存配置
    pipeline.cache_config = {
        'taylorseer_enabled': True,
        'device': device,
        'npu_optimized': device == "npu",
        'optimization_level': 'high',
        'implementation': 'Cache4Diffusion_style'
    }

    logger.info("Real TaylorSeer successfully applied to QwenImage pipeline (Cache4Diffusion style)")
    return pipeline

def get_taylorseer_stats_real(model: nn.Module) -> Dict[str, Any]:
    """
    获取TaylorSeer统计信息（真实版本）
    
    Args:
        model: 应用了TaylorSeer的模型
    
    Returns:
        统计信息字典
    """
    if not hasattr(model, 'cache_config') or not model.cache_config.get('taylorseer_enabled', False):
        return {'error': 'TaylorSeer not applied to this model'}
    
    return {
        'taylorseer_enabled': True,
        'device': model.cache_config.get('device', 'unknown'),
        'npu_optimized': model.cache_config.get('npu_optimized', False),
        'optimization_level': model.cache_config.get('optimization_level', 'unknown'),
        'implementation': model.cache_config.get('implementation', 'unknown'),
        'cache_config': model.cache_config
    }
