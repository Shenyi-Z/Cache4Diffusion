import torch
from typing import Dict, Any, Tuple
from .taylor_calculator import FeatureCacheManager

def cache_init(self, num_layers: int = 24, num_attention_heads: int = 16) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    初始化TaylorSeer缓存系统
    
    Args:
        self: QwenImage模型实例
        num_layers: Transformer层数，QwenImage-1.5默认为24层
        num_attention_heads: 注意力头数，QwenImage-1.5默认为16头
    
    Returns:
        cache_dic: 缓存字典
        current: 当前状态信息
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    
    # 初始化基础缓存结构
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index['layer_index'] = {}
    
    # 注意力图缓存
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    
    # QwenImage使用统一的transformer结构，不需要区分double/single stream
    cache[-1]['transformer'] = {}
    cache_index[-1]['transformer'] = {}
    cache_dic['attn_map'][-1]['transformer'] = {}
    
    # 为每一层初始化缓存
    for j in range(num_layers):
        cache[-1]['transformer'][j] = {}
        cache_index[-1]['transformer'][j] = {}
        cache_dic['attn_map'][-1]['transformer'][j] = {}
        
        # 为每个模块初始化缓存
        cache[-1]['transformer'][j]['self_attn'] = {}
        cache[-1]['transformer'][j]['cross_attn'] = {}
        cache[-1]['transformer'][j]['mlp'] = {}
        cache[-1]['transformer'][j]['norm'] = {}
        
        # 注意力图缓存
        cache_dic['attn_map'][-1]['transformer'][j]['self_attn'] = {}
        cache_dic['attn_map'][-1]['transformer'][j]['cross_attn'] = {}
    
    # 缓存计数器
    cache_dic['cache_counter'] = 0
    
    # TaylorSeer配置
    cache_dic['taylor_cache'] = True
    cache_dic['cache_type'] = 'random'
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    
    # 新鲜度调度配置
    cache_dic['fresh_ratio_schedule'] = 'ToCa'
    cache_dic['fresh_ratio'] = 0.0
    cache_dic['fresh_threshold'] = 6
    cache_dic['force_fresh'] = 'global'
    cache_dic['soft_fresh_weight'] = 0.0
    
    # 泰勒展开配置
    cache_dic['max_order'] = 2  # 最大泰勒阶数（提升到2阶）
    cache_dic['first_enhance'] = 3  # 首次增强步数
    
    # 特征缓存管理器
    cache_dic['feature_cache_manager'] = FeatureCacheManager(max_cache_size=4096)
    
    # NPU特定配置
    cache_dic['npu_optimized'] = True
    cache_dic['memory_efficient'] = True
    cache_dic['precision'] = 'fp16'  # 默认使用FP16精度
    
    # 当前状态信息
    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = getattr(self, 'num_steps', 50)
    current['stream'] = 'transformer'
    current['layer'] = 0
    current['module'] = 'self_attn'
    
    return cache_dic, current

def get_cache_config() -> Dict[str, Any]:
    """
    获取默认缓存配置
    
    Returns:
        默认配置字典
    """
    return {
        'cache_type': 'random',
        'fresh_ratio_schedule': 'ToCa',
        'fresh_threshold': 6,
        'max_order': 1,
        'first_enhance': 3,
        'npu_optimized': True,
        'memory_efficient': True,
        'precision': 'fp16'
    }

