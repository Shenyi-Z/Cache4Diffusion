import torch
from typing import Dict, Any, Optional
from .memory_manager import NPUMemoryManager

class NPUOptimizer:
    """NPU优化器类"""
    
    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.memory_manager = NPUMemoryManager(device)
        self.optimization_level = "medium"
        self.performance_metrics = {}
        
    def set_optimization_level(self, level: str):
        """设置优化级别"""
        valid_levels = ["low", "medium", "high"]
        if level in valid_levels:
            self.optimization_level = level
            return True
        return False
    
    def optimize_memory(self):
        """优化内存使用"""
        return self.memory_manager.optimize_memory()
    
    def clear_cache(self):
        """清理缓存"""
        return self.memory_manager.clear_cache()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            memory_info = self.memory_manager.get_memory_info()
            
            # 获取NPU设备信息
            device_info = {}
            if hasattr(torch, 'npu') and torch.npu.is_available():
                device_info = {
                    "device_count": torch.npu.device_count(),
                    "current_device": torch.npu.current_device(),
                    "device_name": f"npu:{torch.npu.current_device()}"
                }
            
            self.performance_metrics = {
                "optimization_level": self.optimization_level,
                "memory_info": memory_info,
                "device_info": device_info,
                "status": "ready"
            }
            
            return self.performance_metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def apply_optimizations(self):
        """应用优化设置"""
        try:
            # 根据优化级别应用不同的优化策略
            if self.optimization_level == "high":
                # 高级优化
                self.clear_cache()
                if hasattr(torch, 'npu'):
                    torch.npu.empty_cache()
                    
            elif self.optimization_level == "medium":
                # 中等优化
                self.clear_cache()
                
            # 低级别优化 - 不做额外操作
            
            return True
            
        except Exception as e:
            print(f"优化应用失败: {e}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            "level": self.optimization_level,
            "applied": self.apply_optimizations(),
            "metrics": self.get_performance_metrics()
        }
