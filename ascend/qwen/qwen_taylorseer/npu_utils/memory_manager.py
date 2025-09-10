import torch
from typing import Optional, Dict, Any

class NPUMemoryManager:
    """NPU内存管理器"""
    
    def __init__(self, device: str = "npu:0"):
        self.device = device
        self.memory_stats = {}
        
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                allocated = torch.npu.memory_allocated(self.device)
                reserved = torch.npu.memory_reserved(self.device)
                return {
                    "allocated": allocated,
                    "reserved": reserved,
                    "free": reserved - allocated
                }
            else:
                return {"error": "NPU not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def clear_cache(self):
        """清理缓存"""
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.empty_cache()
                return True
            return False
        except Exception:
            return False
    
    def optimize_memory(self):
        """内存优化"""
        self.clear_cache()
        return self.get_memory_info()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "memory_info": self.get_memory_info(),
            "device": self.device,
            "status": "ready"
        }
