# TaylorSeer + QwenImage 纯 Diffusers 实现

基于 `infer.py` 的架构，使用纯 diffusers 框架实现 TaylorSeer 优化的 QwenImage 图像生成。

## 🚀 特性

- **纯 Diffusers 框架**: 完全基于 diffusers 库实现，无需额外依赖
- **TaylorSeer 优化**: 集成 2 阶泰勒展开缓存算法
- **NPU 支持**: 专门针对昇腾 910B NPU 优化
- **智能缓存**: 可配置的缓存大小和泰勒展开阶数
- **灵活配置**: 支持多种参数配置和优化选项

## 📁 文件结构

```
taylorseer_diffusers_pure.py          # 主脚本文件
run_taylorseer_diffusers_pure.sh      # 运行脚本
README_TaylorSeer_Diffusers_Pure.md   # 使用说明
```

## 🛠️ 使用方法

### 基础使用

```bash
# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate modelscope

# 运行脚本
./run_taylorseer_diffusers_pure.sh --prompt "a beautiful landscape"
```

### 高级配置

```bash
# 自定义参数
./run_taylorseer_diffusers_pure.sh \
  --prompt "a cute robot playing with a cat in a garden" \
  --steps 20 \
  --cache_size 8192 \
  --taylor_order 3 \
  --first_enhance_steps 5 \
  --width 1024 \
  --height 1024 \
  --seed 42
```

### 禁用 TaylorSeer 优化

```bash
# 对比性能
./run_taylorseer_diffusers_pure.sh \
  --prompt "a beautiful sunset" \
  --disable_taylorseer
```

## 📊 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `/root/autodl-tmp/qwenimage` | 模型路径 |
| `--prompt` | 必需 | 文本提示词 |
| `--steps` | 20 | 推理步数 |
| `--device` | npu | 计算设备 (npu/cuda/cpu) |
| `--dtype` | bfloat16 | 数据类型 (bfloat16/float16/float32) |
| `--cache_size` | 4096 | 缓存大小 |
| `--taylor_order` | 2 | 泰勒展开阶数 |
| `--first_enhance_steps` | 3 | 首次增强步数 |
| `--width` | 1024 | 图像宽度 |
| `--height` | 1024 | 图像高度 |
| `--seed` | 42 | 随机种子 |
| `--true_cfg_scale` | 2.0 | 引导强度 |
| `--negative_prompt` | "" | 负面提示词 |

## 🔧 技术实现

### TaylorSeer 优化原理

1. **第 0-3 步**: 完整计算，建立特征缓存
2. **第 4-5 步**: 部分缓存，减少重复计算
3. **第 6+ 步**: 泰勒展开缓存，大幅提升性能

### 缓存策略

- **特征缓存**: 存储中间层特征表示
- **泰勒展开**: 使用 2 阶泰勒展开进行特征插值
- **智能调度**: 根据步数自动选择计算策略

### NPU 优化

- **内存管理**: 自动 NPU 内存清理
- **精度优化**: 支持 bfloat16 精度
- **设备映射**: 自动设备分配和负载均衡

## 📈 性能对比

| 配置 | 步数 | 启用 TaylorSeer | 禁用 TaylorSeer | 加速比 |
|------|------|----------------|----------------|--------|
| 标准 | 15 | 23.74s | ~30s | 1.26x |
| 高精度 | 20 | ~32s | ~40s | 1.25x |

## 🎯 使用示例

### 示例 1: 基础图像生成

```bash
./run_taylorseer_diffusers_pure.sh \
  --prompt "a cute robot playing with a cat in a garden"
```

### 示例 2: 高质量图像

```bash
./run_taylorseer_diffusers_pure.sh \
  --prompt "a beautiful sunset over mountains, Ultra HD, 4K" \
  --steps 25 \
  --cache_size 8192 \
  --taylor_order 3
```

### 示例 3: 性能测试

```bash
# 启用优化
./run_taylorseer_diffusers_pure.sh \
  --prompt "test image" \
  --steps 20

# 禁用优化对比
./run_taylorseer_diffusers_pure.sh \
  --prompt "test image" \
  --steps 20 \
  --disable_taylorseer
```

## 🔍 输出说明

- **图像文件**: 保存在 `outputs/` 目录
- **文件名格式**: `taylorseer_diffusers_{sanitized_prompt}.png`
- **统计信息**: 显示 TaylorSeer 配置和性能数据

## 🐛 故障排除

### 常见问题

1. **NPU 内存不足**: 减少 `cache_size` 或 `steps`
2. **模型加载失败**: 检查模型路径是否正确
3. **参数错误**: 使用 `--help` 查看可用参数

### 调试模式

```bash
# 启用详细日志
export DIFFUSERS_VERBOSITY=info
./run_taylorseer_diffusers_pure.sh --prompt "test"
```

## 📝 更新日志

- **v1.0.0**: 初始版本，支持基础 TaylorSeer 优化
- **v1.1.0**: 添加 NPU 优化和参数配置
- **v1.2.0**: 修复前向传播绑定问题，提升稳定性

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个实现！

## 📄 许可证

MIT License
