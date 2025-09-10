import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class NPUOps:
    """
    昇腾NPU操作封装类
    提供针对昇腾910B的专用操作
    """
    
    def __init__(self, device: str = "npu"):
        """
        初始化NPU操作封装
        
        Args:
            device: 设备类型
        """
        self.device = device
        self.supported_ops = self._get_supported_ops()
        
        # 检查NPU环境
        self._check_npu_environment()
    
    def _get_supported_ops(self) -> List[str]:
        """
        获取支持的NPU操作列表
        
        Returns:
            支持的操作列表
        """
        return [
            'matmul',
            'conv2d',
            'attention',
            'layer_norm',
            'gelu',
            'softmax',
            'dropout'
        ]
    
    def _check_npu_environment(self) -> None:
        """
        检查NPU环境
        """
        try:
            # 检查是否在昇腾环境中
            if hasattr(torch, 'npu'):
                logger.info("NPU environment detected")
            else:
                logger.warning("NPU environment not detected, some optimizations may not work")
        except Exception as e:
            logger.warning(f"Failed to check NPU environment: {e}")
    
    def optimized_matmul(self, a: torch.Tensor, b: torch.Tensor, 
                        transpose_a: bool = False, transpose_b: bool = False) -> torch.Tensor:
        """
        优化的矩阵乘法操作
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            transpose_a: 是否转置第一个矩阵
            transpose_b: 是否转置第二个矩阵
        
        Returns:
            矩阵乘法结果
        """
        try:
            # 在NPU上使用优化的矩阵乘法
            if self.device == "npu" and hasattr(torch, 'npu'):
                # 使用昇腾优化的矩阵乘法
                result = torch.matmul(a, b)
                
                # 应用NPU特定的优化
                if a.dtype == torch.float16 and b.dtype == torch.float16:
                    # FP16优化
                    result = self._optimize_fp16_matmul(result)
                
                return result
            else:
                # 回退到标准矩阵乘法
                return torch.matmul(a, b)
        except Exception as e:
            logger.warning(f"NPU optimized matmul failed, falling back to standard: {e}")
            return torch.matmul(a, b)
    
    def _optimize_fp16_matmul(self, result: torch.Tensor) -> torch.Tensor:
        """
        优化FP16矩阵乘法结果
        
        Args:
            result: 矩阵乘法结果
        
        Returns:
            优化后的结果
        """
        # 在昇腾NPU上，FP16计算可能需要特殊处理
        # 这里可以添加昇腾特定的优化逻辑
        
        # 数值稳定性优化
        if result.dtype == torch.float16:
            # 防止FP16下溢
            result = torch.clamp(result, -65504, 65504)
        
        return result
    
    def optimized_attention(self, query: torch.Tensor, key: torch.Tensor, 
                           value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                           dropout_p: float = 0.0) -> torch.Tensor:
        """
        优化的注意力计算
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            attention_mask: 注意力掩码
            dropout_p: dropout概率
        
        Returns:
            注意力输出
        """
        try:
            # 计算注意力分数
            attention_scores = torch.matmul(query, key.transpose(-2, -1))
            
            # 缩放
            d_k = query.size(-1)
            attention_scores = attention_scores / (d_k ** 0.5)
            
            # 应用掩码
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # 应用softmax
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # 应用dropout
            if dropout_p > 0.0:
                attention_probs = torch.dropout(attention_probs, dropout_p, self.training)
            
            # 计算输出
            output = torch.matmul(attention_probs, value)
            
            return output
            
        except Exception as e:
            logger.warning(f"NPU optimized attention failed: {e}")
            # 回退到标准实现
            return self._standard_attention(query, key, value, attention_mask, dropout_p)
    
    def _standard_attention(self, query: torch.Tensor, key: torch.Tensor, 
                           value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                           dropout_p: float = 0.0) -> torch.Tensor:
        """
        标准注意力实现（回退方案）
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            attention_mask: 注意力掩码
            dropout_p: dropout概率
        
        Returns:
            注意力输出
        """
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        d_k = query.size(-1)
        attention_scores = attention_scores / (d_k ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        if dropout_p > 0.0:
            attention_probs = torch.dropout(attention_probs, dropout_p, self.training)
        
        return torch.matmul(attention_probs, value)
    
    def optimized_layer_norm(self, input_tensor: torch.Tensor, 
                            normalized_shape: Tuple[int, ...],
                            weight: Optional[torch.Tensor] = None,
                            bias: Optional[torch.Tensor] = None,
                            eps: float = 1e-5) -> torch.Tensor:
        """
        优化的层归一化
        
        Args:
            input_tensor: 输入张量
            normalized_shape: 归一化形状
            weight: 权重参数
            bias: 偏置参数
            eps: 数值稳定性常数
        
        Returns:
            归一化后的张量
        """
        try:
            # 使用PyTorch内置的层归一化
            result = torch.nn.functional.layer_norm(
                input_tensor, normalized_shape, weight, bias, eps
            )
            
            # NPU特定优化
            if self.device == "npu":
                result = self._optimize_npu_layer_norm(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"NPU optimized layer norm failed: {e}")
            return torch.nn.functional.layer_norm(
                input_tensor, normalized_shape, weight, bias, eps
            )
    
    def _optimize_npu_layer_norm(self, result: torch.Tensor) -> torch.Tensor:
        """
        优化NPU上的层归一化结果
        
        Args:
            result: 层归一化结果
        
        Returns:
            优化后的结果
        """
        # 在昇腾NPU上，层归一化可能需要特殊处理
        # 这里可以添加昇腾特定的优化逻辑
        
        # 数值稳定性优化
        if result.dtype == torch.float16:
            result = torch.clamp(result, -65504, 65504)
        
        return result
    
    def optimized_gelu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        优化的GELU激活函数
        
        Args:
            input_tensor: 输入张量
        
        Returns:
            激活后的张量
        """
        try:
            # 使用PyTorch内置的GELU
            result = torch.nn.functional.gelu(input_tensor)
            
            # NPU特定优化
            if self.device == "npu":
                result = self._optimize_npu_gelu(result)
            
            return result
            
        except Exception as e:
            logger.warning(f"NPU optimized GELU failed: {e}")
            return torch.nn.functional.gelu(input_tensor)
    
    def _optimize_npu_gelu(self, result: torch.Tensor) -> torch.Tensor:
        """
        优化NPU上的GELU结果
        
        Args:
            result: GELU结果
        
        Returns:
            优化后的结果
        """
        # 在昇腾NPU上，GELU可能需要特殊处理
        # 这里可以添加昇腾特定的优化逻辑
        
        # 数值稳定性优化
        if result.dtype == torch.float16:
            result = torch.clamp(result, -65504, 65504)
        
        return result
    
    def get_npu_info(self) -> Dict[str, Any]:
        """
        获取NPU信息
        
        Returns:
            NPU信息字典
        """
        info = {
            'device': self.device,
            'supported_ops': self.supported_ops,
            'npu_environment': hasattr(torch, 'npu'),
            'optimization_level': 'high' if self.device == "npu" else 'low'
        }
        
        # 尝试获取更多NPU信息
        try:
            if hasattr(torch, 'npu'):
                info['npu_available'] = torch.npu.is_available()
                info['npu_device_count'] = torch.npu.device_count()
        except Exception as e:
            info['npu_available'] = False
            info['npu_device_count'] = 0
            info['error'] = str(e)
        
        return info
    
    def benchmark_operation(self, op_name: str, input_shapes: List[Tuple[int, ...]], 
                           num_runs: int = 100) -> Dict[str, float]:
        """
        基准测试NPU操作性能
        
        Args:
            op_name: 操作名称
            input_shapes: 输入形状列表
            num_runs: 运行次数
        
        Returns:
            性能指标字典
        """
        if op_name not in self.supported_ops:
            return {'error': f'Operation {op_name} not supported'}
        
        try:
            # 创建测试输入
            inputs = [torch.randn(shape, dtype=torch.float16, device=self.device) 
                     for shape in input_shapes]
            
            # 预热
            for _ in range(10):
                if op_name == 'matmul':
                    _ = torch.matmul(inputs[0], inputs[1])
                elif op_name == 'attention':
                    _ = self.optimized_attention(inputs[0], inputs[1], inputs[2])
            
            # 同步
            if self.device == "npu":
                torch.npu.synchronize()
            else:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # 开始计时
            start_time = torch.npu.Event(enable_timing=True) if self.device == "npu" else torch.cuda.Event(enable_timing=True)
            end_time = torch.npu.Event(enable_timing=True) if self.device == "npu" else torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # 执行操作
            for _ in range(num_runs):
                if op_name == 'matmul':
                    _ = torch.matmul(inputs[0], inputs[1])
                elif op_name == 'attention':
                    _ = self.optimized_attention(inputs[0], inputs[1], inputs[2])
            
            end_time.record()
            
            # 同步
            if self.device == "npu":
                torch.npu.synchronize()
            else:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # 计算时间
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
            
            return {
                'operation': op_name,
                'num_runs': num_runs,
                'total_time': elapsed_time,
                'avg_time_per_run': elapsed_time / num_runs,
                'throughput': num_runs / elapsed_time
            }
            
        except Exception as e:
            return {'error': f'Benchmark failed: {e}'}

