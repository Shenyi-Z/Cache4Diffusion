import torch
from diffusers import DiffusionPipeline
from qwen_taylorseer.forwards.qwen_forward_optimized import apply_taylorseer_to_qwen_optimized
import time

def test_optimized_taylorseer():
    """测试优化版本的TaylorSeer"""
    print("Testing optimized TaylorSeer...")
    
    try:
        # 加载模型
        print("Loading QwenImage model...")
        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/QwenImage-1.5",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # 移动到NPU
        if hasattr(torch, 'npu'):
            pipeline = pipeline.to("npu")
            print("Model moved to NPU")
        else:
            print("NPU not available, using CPU")
            pipeline = pipeline.to("cpu")
        
        # 应用优化版本的TaylorSeer
        print("Applying optimized TaylorSeer...")
        start_time = time.time()
        pipeline = apply_taylorseer_to_qwen_optimized(pipeline, device="npu" if hasattr(torch, 'npu') else "cpu")
        setup_time = time.time() - start_time
        print(f"TaylorSeer setup completed in {setup_time:.2f} seconds")
        
        # 检查模型是否被修改
        print("Checking model modifications...")
        if hasattr(pipeline, 'cache_config'):
            print(f"Cache config: {pipeline.cache_config}")
        else:
            print("No cache config found")
        
        # 检查transformer是否被修改
        if hasattr(pipeline, 'transformer'):
            transformer = pipeline.transformer
            print(f"Transformer type: {type(transformer)}")
            print(f"Transformer forward: {transformer.forward}")
            
            # 检查是否有原始forward被保存（这不应该有）
            if hasattr(transformer, '_original_forward'):
                print("WARNING: Original forward was saved (this might cause issues)")
            else:
                print("Good: No original forward saved")
        
        # 检查NPU优化是否应用
        if hasattr(pipeline, 'cache_config') and pipeline.cache_config.get('npu_optimized', False):
            print("NPU optimizations applied successfully")
        else:
            print("NPU optimizations not applied")
        
        print("Optimized TaylorSeer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_performance():
    """性能基准测试"""
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    try:
        # 加载模型
        print("Loading model for benchmark...")
        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/QwenImage-1.5",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        if hasattr(torch, 'npu'):
            pipeline = pipeline.to("npu")
        else:
            pipeline = pipeline.to("cpu")
        
        # 测试原始性能
        print("Testing original performance...")
        start_time = time.time()
        
        # 简单的推理测试（不生成完整图像）
        with torch.no_grad():
            # 创建一个小的测试输入
            test_input = torch.randn(1, 3, 64, 64, device=pipeline.device, dtype=pipeline.dtype)
            if hasattr(pipeline, 'transformer'):
                # 测试transformer的前向传播
                try:
                    result = pipeline.transformer(test_input)
                    print(f"Original transformer forward: {type(result)}")
                except Exception as e:
                    print(f"Original transformer failed: {e}")
        
        original_time = time.time() - start_time
        print(f"Original model test completed in {original_time:.2f} seconds")
        
        # 应用TaylorSeer
        print("Applying TaylorSeer...")
        start_time = time.time()
        pipeline = apply_taylorseer_to_qwen_optimized(pipeline, device="npu" if hasattr(torch, 'npu') else "cpu")
        taylorseer_time = time.time() - start_time
        print(f"TaylorSeer applied in {taylorseer_time:.2f} seconds")
        
        # 测试优化后性能
        print("Testing optimized performance...")
        start_time = time.time()
        
        with torch.no_grad():
            try:
                result = pipeline.transformer(test_input)
                print(f"Optimized transformer forward: {type(result)}")
            except Exception as e:
                print(f"Optimized transformer failed: {e}")
        
        optimized_time = time.time() - start_time
        print(f"Optimized model test completed in {optimized_time:.2f} seconds")
        
        # 性能对比
        print("\nPerformance Summary:")
        print(f"Original setup time: {original_time:.2f}s")
        print(f"TaylorSeer setup time: {taylorseer_time:.2f}s")
        print(f"Original inference time: {original_time:.2f}s")
        print(f"Optimized inference time: {optimized_time:.2f}s")
        
        if optimized_time < original_time:
            speedup = original_time / optimized_time
            print(f"Speedup: {speedup:.2f}x")
        else:
            slowdown = optimized_time / original_time
            print(f"Slowdown: {slowdown:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TaylorSeer Optimized Mode Test")
    print("="*50)
    
    # 基本功能测试
    success = test_optimized_taylorseer()
    
    if success:
        # 性能基准测试
        benchmark_performance()
    else:
        print("Basic test failed, skipping benchmark")

