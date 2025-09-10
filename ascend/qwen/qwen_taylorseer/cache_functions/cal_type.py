import torch
from typing import Dict, Any

def cal_type(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    """
    计算当前步骤的缓存类型
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息
    """
    step = current['step']
    num_steps = current['num_steps']
    
    # 计算缓存类型
    if step == 0:
        # 第一步：完整计算
        current['type'] = 'full'
        current['cache_hit'] = False
    elif step < cache_dic['first_enhance']:
        # 前几步：使用缓存但可能不完整
        current['type'] = 'partial'
        current['cache_hit'] = True
    else:
        # 后续步骤：使用泰勒展开缓存
        current['type'] = 'Taylor'
        current['cache_hit'] = True
    
    # 更新激活步骤列表
    if step not in current['activated_steps']:
        current['activated_steps'].append(step)
    
    # 计算新鲜度
    if len(current['activated_steps']) >= 2:
        step_diff = current['activated_steps'][-1] - current['activated_steps'][-2]
        current['step_diff'] = step_diff
    else:
        current['step_diff'] = 1
    
    # NPU特定优化
    if cache_dic.get('npu_optimized', False):
        _optimize_for_npu(cache_dic, current)

def _optimize_for_npu(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    """
    针对昇腾NPU的优化
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息
    """
    # 内存效率优化
    if cache_dic.get('memory_efficient', False):
        current['use_mixed_precision'] = True
        current['gradient_checkpointing'] = False  # NPU上通常不需要
    
    # 精度优化
    precision = cache_dic.get('precision', 'fp16')
    if precision == 'fp16':
        current['dtype'] = torch.float16
    elif precision == 'bf16':
        current['dtype'] = torch.bfloat16
    else:
        current['dtype'] = torch.float32
    
    # 缓存大小自适应调整
    if current['type'] == 'Taylor':
        # 泰勒模式：减少缓存大小以节省内存
        current['cache_size_limit'] = 1024 * 1024  # 1MB
    else:
        # 完整模式：允许更大的缓存
        current['cache_size_limit'] = 4 * 1024 * 1024  # 4MB

def get_cache_efficiency(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> float:
    """
    计算缓存效率
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息
    
    Returns:
        缓存效率 (0.0 - 1.0)
    """
    if not current.get('cache_hit', False):
        return 0.0
    
    total_steps = current.get('num_steps', 50)
    cached_steps = len(current.get('activated_steps', []))
    
    if total_steps == 0:
        return 0.0
    
    return min(1.0, cached_steps / total_steps)

def should_clear_cache(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """
    判断是否应该清理缓存
    
    Args:
        cache_dic: 缓存字典
        current: 当前状态信息
    
    Returns:
        是否应该清理缓存
    """
    # 检查内存使用
    if current.get('cache_size_limit', 0) > 0:
        estimated_size = _estimate_cache_size(cache_dic)
        if estimated_size > current['cache_size_limit']:
            return True
    
    # 检查新鲜度阈值
    if current.get('step_diff', 0) > cache_dic.get('fresh_threshold', 6):
        return True
    
    return False

def _estimate_cache_size(cache_dic: Dict[str, Any]) -> int:
    """
    估算缓存大小（字节）
    
    Args:
        cache_dic: 缓存字典
    
    Returns:
        估算的缓存大小（字节）
    """
    # 这是一个简化的估算，实际实现中需要更精确的计算
    total_size = 0
    
    if 'cache' in cache_dic and -1 in cache_dic['cache']:
        cache = cache_dic['cache'][-1]
        for stream_name, stream_cache in cache.items():
            if isinstance(stream_cache, dict):
                for layer_idx, layer_cache in stream_cache.items():
                    if isinstance(layer_cache, dict):
                        for module_name, module_cache in layer_cache.items():
                            if isinstance(module_cache, dict):
                                # 假设每个特征张量平均为1MB
                                total_size += len(module_cache) * 1024 * 1024
    
    return total_size

