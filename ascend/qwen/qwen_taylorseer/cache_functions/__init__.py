from .cache_init import cache_init
from .cal_type import cal_type
from .npu_optimizer import NPUOptimizer
from .taylor_calculator import TaylorCalculator, FeatureCacheManager

def get_cache_config():
    return {
        'cache_optimization_level': 'medium',
        'npu_optimized': False,
        'memory_efficient': False,
        'precision': 'fp16',
        'cache_counter': 0,
    }

__all__ = [
    "cache_init",
    "cal_type", 
    "NPUOptimizer",
    "TaylorCalculator",
    "FeatureCacheManager",
    "get_cache_config",
]

