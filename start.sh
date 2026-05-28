#!/bin/bash

source .venv/bin/activate

export HTTP_PROXY="http://127.0.0.1:1080"
export HTTPS_PROXY="http://127.0.0.1:1080"

# 设置 CUDA 设备
#export CUDA_VISIBLE_DEVICES=$DEVICE
export CUDNN_V8_API_ENABLED=1
export UV_LINK_MODE=copy
export COMFYUI_ROOT=$(pwd)
echo "COMFYUI_ROOT=$COMFYUI_ROOT"

TORCHINDUCTOR_FREEZING=1

let PORT=8888
USER="user_$PORT"
HOME="/data/ai/comfyui-user"
DB="sqlite:///$HOME/user/$USER.db"
echo "home=$HOME"
echo "database=$DB"

#--use-flash-attention
#--use-sage-attention
# --cache-lru 3
#--force-fp16 
#--bf16-vae
python main.py \
	--port $PORT \
	--listen 0.0.0.0 \
	--enable-manager \
	--enable-cors-header \
	--fp8_e4m3fn-unet \
	--fp8_e4m3fn-text-enc \
	--supports-fp8-compute \
	--force-channels-last \
        --enable-triton-backend \
        --enable-dynamic-vram \
	--normalvram \
	--fast fp16_accumulation fp8_matrix_mult cublas_ops autotune \
	--use-sage-attention \
	--mmap-torch-files \
	--cache-lru 2 \
        --multi-user \
	--database-url "$DB"	\
	--reserve-vram 0.5 \
	--async-offload \
	--base-directory $HOME \
	--output-directory /data/share/comfyui-output-$USER

