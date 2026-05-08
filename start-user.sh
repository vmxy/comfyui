#!/bin/bash

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
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

# 激活虚拟环境
source .venv/bin/activate

export HTTP_PROXY="http://127.0.0.1:1080"
export HTTPS_PROXY="http://127.0.0.1:1080"

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$DEVICE
export CUDNN_V8_API_ENABLED=1
export UV_LINK_MODE=copy

let PORT=9900+DEVICE
USER="user_$PORT"
HOME="/data/ai/comfyui-user"
DB="sqlite:///$HOME/user/$USER.db"
echo "PORT=$PORT DEVICES=$DEVICE USRER=$USER"
echo "home=$HOME"
echo "database=$DB"

#--use-flash-attention
#--use-sage-attention
# --cache-lru 3
#
python main.py \
	--cuda-device $DEVICE \
	--port $PORT \
	--listen 0.0.0.0 \
	--enable-manager \
	--enable-cors-header \
	--force-fp16 \
	--fp8_e4m3fn-unet \
	--bf16-vae \
	--fp8_e4m3fn-text-enc \
	--supports-fp8-compute \
	--normalvram \
	--fast fp16_accumulation fp8_matrix_mult cublas_ops autotune \
	--use-flash-attention \
	--mmap-torch-files \
	--cache-lru 2 \
        --multi-user \
	--database-url "$DB"	\
	--reserve-vram 0.5 \
	--async-offload \
	--base-directory $HOME \
	--output-directory /data/share/comfyui-output-$USER

