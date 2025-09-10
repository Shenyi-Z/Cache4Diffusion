#!/bin/bash

# TaylorSeer + QwenImage 纯 Diffusers 实现运行脚本
# 基于 infer.py 的架构，集成 TaylorSeer 优化

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate modelscope

# 默认参数
MODEL="/root/autodl-tmp/qwenimage"
PROMPT="a beautiful landscape with mountains and lake, Ultra HD, 4K, cinematic composition"
STEPS=20
DEVICE="npu"
DTYPE="bfloat16"
CACHE_SIZE=4096
TAYLOR_ORDER=2
FIRST_ENHANCE_STEPS=3

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --cache_size)
            CACHE_SIZE="$2"
            shift 2
            ;;
        --taylor_order)
            TAYLOR_ORDER="$2"
            shift 2
            ;;
        --first_enhance_steps)
            FIRST_ENHANCE_STEPS="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --cfg_scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --negative_prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --disable_taylorseer)
            DISABLE_TAYLORSEER="--no-enable_taylorseer"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "可用参数:"
            echo "  --model MODEL_PATH          模型路径 (默认: /root/autodl-tmp/qwenimage)"
            echo "  --prompt PROMPT             提示词"
            echo "  --steps STEPS               推理步数 (默认: 20)"
            echo "  --device DEVICE             设备 (默认: npu)"
            echo "  --dtype DTYPE               数据类型 (默认: bfloat16)"
            echo "  --cache_size SIZE           缓存大小 (默认: 4096)"
            echo "  --taylor_order ORDER        泰勒展开阶数 (默认: 2)"
            echo "  --first_enhance_steps STEPS 首次增强步数 (默认: 3)"
            echo "  --width WIDTH               图像宽度 (默认: 1024)"
            echo "  --height HEIGHT             图像高度 (默认: 1024)"
            echo "  --seed SEED                 随机种子 (默认: 42)"
            echo "  --cfg_scale SCALE           引导强度 (默认: 2.0)"
            echo "  --negative_prompt PROMPT    负面提示词"
            echo "  --disable_taylorseer        禁用 TaylorSeer 优化"
            exit 1
            ;;
    esac
done

echo "🚀 启动 TaylorSeer + QwenImage (纯 Diffusers 实现)..."
echo "模型: $MODEL"
echo "提示词: $PROMPT"
echo "步数: $STEPS"
echo "设备: $DEVICE"
echo "精度: $DTYPE"
echo "缓存大小: $CACHE_SIZE"
echo "泰勒阶数: $TAYLOR_ORDER"
echo "首次增强步数: $FIRST_ENHANCE_STEPS"
echo "架构: 纯 Diffusers 框架"
echo "优化: TaylorSeer 缓存算法"

# 创建输出目录
mkdir -p outputs

# 构建命令参数
CMD_ARGS="--model \"$MODEL\" --prompt \"$PROMPT\" --steps $STEPS --device \"$DEVICE\" --dtype \"$DTYPE\""
CMD_ARGS="$CMD_ARGS --cache_size $CACHE_SIZE --taylor_order $TAYLOR_ORDER --first_enhance_steps $FIRST_ENHANCE_STEPS"

if [ ! -z "$WIDTH" ]; then
    CMD_ARGS="$CMD_ARGS --width $WIDTH"
fi

if [ ! -z "$HEIGHT" ]; then
    CMD_ARGS="$CMD_ARGS --height $HEIGHT"
fi

if [ ! -z "$SEED" ]; then
    CMD_ARGS="$CMD_ARGS --seed $SEED"
fi

if [ ! -z "$CFG_SCALE" ]; then
    CMD_ARGS="$CMD_ARGS --true_cfg_scale $CFG_SCALE"
fi

if [ ! -z "$NEGATIVE_PROMPT" ]; then
    CMD_ARGS="$CMD_ARGS --negative_prompt \"$NEGATIVE_PROMPT\""
fi

if [ ! -z "$DISABLE_TAYLORSEER" ]; then
    CMD_ARGS="$CMD_ARGS $DISABLE_TAYLORSEER"
fi

# 运行推理（切到脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "执行命令: python3 $SCRIPT_DIR/taylorseer_diffusers_pure.py $CMD_ARGS"
eval "python3 $SCRIPT_DIR/taylorseer_diffusers_pure.py $CMD_ARGS"

echo ""
echo "🎉 推理完成！"
echo "输出目录: outputs/"