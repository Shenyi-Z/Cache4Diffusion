import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TaylorCalculator:
    """泰勒展开计算器"""
    
    def __init__(self, max_order: int = 2, cache_size: int = 1024):
        self.max_order = max_order
        self.cache_size = cache_size
        self.feature_cache = {}
        self.gradient_cache = {}
        
    def compute_taylor_expansion(self, 
                                current_features: torch.Tensor,
                                cached_features: torch.Tensor,
                                step_diff: int,
                                order: int = 1) -> torch.Tensor:
        """
        计算泰勒展开
        
        Args:
            current_features: 当前步骤的特征
            cached_features: 缓存的特征
            step_diff: 步数差
            order: 泰勒展开阶数
        
        Returns:
            泰勒展开后的特征
        """
        if order == 0:
            return cached_features
        
        # 计算一阶导数（梯度）
        if order >= 1:
            gradient = (current_features - cached_features) / max(step_diff, 1)
            
            # 泰勒展开：f(x+h) ≈ f(x) + h*f'(x) + (h²/2)*f''(x) + ...
            result = cached_features + step_diff * gradient
            
            if order >= 2:
                # 二阶导数近似
                second_derivative = self._estimate_second_derivative(current_features, cached_features, step_diff)
                result += (step_diff ** 2) * second_derivative / 2
                
            return result
        
        return cached_features
    
    def _estimate_second_derivative(self, 
                                   current: torch.Tensor, 
                                   cached: torch.Tensor, 
                                   step_diff: int) -> torch.Tensor:
        """估计二阶导数"""
        if step_diff <= 1:
            return torch.zeros_like(current)
        
        # 使用有限差分估计二阶导数
        return (current - 2 * cached + cached) / (step_diff ** 2)
    
    def cache_features(self, 
                      step: int, 
                      features: torch.Tensor, 
                      attention_weights: Optional[torch.Tensor] = None):
        """缓存特征和注意力权重"""
        if len(self.feature_cache) >= self.cache_size:
            # 移除最旧的缓存
            oldest_step = min(self.feature_cache.keys())
            del self.feature_cache[oldest_step]
            if oldest_step in self.gradient_cache:
                del self.gradient_cache[oldest_step]
        
        # 缓存特征
        self.feature_cache[step] = features.detach().clone()
        
        # 缓存注意力权重
        if attention_weights is not None:
            if 'attention' not in self.feature_cache:
                self.feature_cache['attention'] = {}
            self.feature_cache['attention'][step] = attention_weights.detach().clone()
    
    def get_cached_features(self, step: int) -> Optional[torch.Tensor]:
        """获取缓存的特征"""
        return self.feature_cache.get(step, None)
    
    def compute_gradient(self, current_step: int, previous_step: int) -> Optional[torch.Tensor]:
        """计算两个步骤之间的梯度"""
        if current_step not in self.feature_cache or previous_step not in self.feature_cache:
            return None
        
        current_features = self.feature_cache[current_step]
        previous_features = self.feature_cache[previous_step]
        
        return (current_features - previous_features) / (current_step - previous_step)

class FeatureCacheManager:
    """特征缓存管理器"""
    
    def __init__(self, max_cache_size: int = 2048):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_count = {}
        self.taylor_calculator = TaylorCalculator()
        
    def store_features(self, 
                      step: int, 
                      layer: int, 
                      module: str, 
                      features: torch.Tensor,
                      attention_weights: Optional[torch.Tensor] = None):
        """存储特征"""
        key = f"step_{step}_layer_{layer}_{module}"
        
        # 检查缓存大小
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        # 存储特征
        self.cache[key] = {
            'features': features.detach().clone(),
            'attention_weights': attention_weights.detach().clone() if attention_weights is not None else None,
            'step': step,
            'layer': layer,
            'module': module,
            'timestamp': None  # 简化时间戳，避免设备兼容性问题
        }
        
        self.access_count[key] = 0
    
    def retrieve_features(self, 
                         step: int, 
                         layer: int, 
                         module: str,
                         use_taylor: bool = True,
                         taylor_order: int = 1) -> Optional[torch.Tensor]:
        """检索特征，可选择使用泰勒展开"""
        key = f"step_{step}_layer_{layer}_{module}"
        
        if key in self.cache:
            self.access_count[key] += 1
            cached_data = self.cache[key]
            
            if use_taylor and taylor_order > 0:
                # 使用泰勒展开
                return self._apply_taylor_expansion(cached_data, step, taylor_order)
            else:
                return cached_data['features']
        
        return None
    
    def _apply_taylor_expansion(self, 
                               cached_data: Dict, 
                               current_step: int, 
                               order: int) -> torch.Tensor:
        """应用泰勒展开"""
        cached_step = cached_data['step']
        step_diff = current_step - cached_step
        
        if step_diff <= 0:
            return cached_data['features']
        
        # 寻找最近的缓存步骤进行泰勒展开
        best_cached_step = self._find_best_cached_step(cached_step, current_step)
        if best_cached_step is None:
            return cached_data['features']
        
        best_key = f"step_{best_cached_step}_layer_{cached_data['layer']}_{cached_data['module']}"
        if best_key not in self.cache:
            return cached_data['features']
        
        best_features = self.cache[best_key]['features']
        current_features = cached_data['features']
        
        return self.taylor_calculator.compute_taylor_expansion(
            current_features, best_features, step_diff, order
        )
    
    def _find_best_cached_step(self, cached_step: int, current_step: int) -> Optional[int]:
        """找到最佳的缓存步骤用于泰勒展开"""
        available_steps = [k for k in self.cache.keys() if k.startswith(f"step_")]
        if not available_steps:
            return None
        
        # 选择最接近的缓存步骤
        step_numbers = [int(k.split('_')[1]) for k in available_steps]
        step_numbers.sort()
        
        # 找到小于current_step的最大步骤
        valid_steps = [s for s in step_numbers if s < current_step]
        if not valid_steps:
            return None
        
        return max(valid_steps)
    
    def _evict_oldest(self):
        """移除最旧的缓存"""
        if not self.cache:
            return
        
        # 基于访问次数和时间戳选择要移除的缓存
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: (self.access_count.get(k, 0), 
                                     self.cache[k].get('step', 0)))
        
        del self.cache[oldest_key]
        if oldest_key in self.access_count:
            del self.access_count[oldest_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_utilization': len(self.cache) / self.max_cache_size,
            'total_accesses': sum(self.access_count.values()),
            'avg_access_per_item': sum(self.access_count.values()) / max(len(self.access_count), 1)
        }
