#!/bin/bash

#network_mode = personal_cloud
echo "请修改 user/__manager/config.ini network_mode = personal_cloud"

CACHE=0
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
    --cache)
            CACHE=1
            shift 1
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --device <gpu_id>"
            exit 1
            ;;
    esac
done

# 检查是否提供了 device 参数
if [ -z "$DEVICE" ]; then
    echo "错误: 必须指定 --device 参数"
    echo "用法: $0 --device <gpu_id>"
    echo "示例: $0 --device 0"
    exit 1
fi

if [ "$CACHE" -eq 1 ]; then
    echo "缓存已启用"
    CACHE_ARGS="--cache-lru 4"
else
    echo "缓存已禁用"
    CACHE_ARGS="--cache-none"
fi
# 激活虚拟环境
source .venv/bin/activate

export UV_CACHE_DIR="/data/ai-code/uv-cache"
export HTTP_PROXY="http://127.0.0.1:1080"
export HTTPS_PROXY="http://127.0.0.1:1080"
export HF_ENDPOINT="https://hf-mirror.com"

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$DEVICE
export CUDNN_V8_API_ENABLED=1
#export UV_LINK_MODE=copy



TORCHINDUCTOR_FREEZING=1

let PORT=9900+DEVICE
USER="user_$PORT"
HOME="/data/ai/comfyui-user"
DB="sqlite:///$HOME/user/$USER.db"
Output="/data/share/$(basename "$PWD")-$PORT"
echo "PORT=$PORT DEVICES=$DEVICE"
mkdir -p $Output

#echo "home=$HOME"
#echo "database=$DB"
echo "output=$Output"
#--use-flash-attention
#--use-sage-attention
#--use-pytorch-cross-attention
# --cache-lru 3
#--force-fp16 
#--bf16-vae
#--cache-lru 2
#--base-directory $HOME 
#--multi-user 
#--database-url "$DB"   
python main.py \
    --cuda-device $DEVICE \
    --port $PORT \
    --listen 0.0.0.0 \
    --enable-manager \
    --enable-cors-header \
    --bf16-unet \
    --bf16-text-enc \
    --supports-fp8-compute \
    --force-channels-last \
    --enable-triton-backend \
    --enable-dynamic-vram \
    --fast fp16_accumulation fp8_matrix_mult cublas_ops autotune \
    --use-pytorch-cross-attention \
    --mmap-torch-files \
    $CACHE_ARGS \
    --reserve-vram 0.5 \
    --async-offload \
    --output-directory $Output

