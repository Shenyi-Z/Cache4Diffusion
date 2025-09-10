"""
NPU工具模块
包含昇腾910B NPU的专用工具和优化器
"""

from .npu_ops import NPUOps
from .memory_manager import NPUMemoryManager
from .npu_optimizer import NPUOptimizer

# 兼容旧命名
NPUCacheManager = NPUMemoryManager

__all__ = [
    "NPUOps",
    "NPUMemoryManager",
    "NPUOptimizer",
    "NPUCacheManager",
]
