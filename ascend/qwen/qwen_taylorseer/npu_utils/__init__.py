from .npu_ops import NPUOps
from .memory_manager import NPUMemoryManager
from .npu_optimizer import NPUOptimizer

NPUCacheManager = NPUMemoryManager

__all__ = [
    "NPUOps",
    "NPUMemoryManager",
    "NPUOptimizer",
    "NPUCacheManager",
]
