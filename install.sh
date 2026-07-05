#!/bin/bash

export UV_CACHE_DIR="/data/ai-code/uv-cache"

## --python=3.11 torch=2.8 有flash-attn=2.8
## --python=3.12 torch=2.9 有flash-attn=2.8
## uv venv --python 3.12 .venv
uv venv --python 3.12 .venv #3.10
#export LD_PRELOAD=~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/libpython3.12.so.1.0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0
TORCH_VERSION=2.11.0 # $env:TORCH_VERSION=2.9.1 set TORCH_VERSION=2.9.1
BUILD_CUDA=cu130


sed -i 's/VIRTUAL_ENV_PROMPT=.*/VIRTUAL_ENV_PROMPT="comfy"/g' .venv/bin/activate
source .venv/bin/activate

# wan2.2-animate使用
#uv pip install moviepy sam2 matplotlib onnxruntime

#uv pip install torch torchaudio torchvision "torch~=$TORCH_VERSION"  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130
#https://download.pytorch.org/whl/cu130 torchvision==0.24.0+${BUILD_CUDA}
TVERSION=${TORCH_VERSION}+${BUILD_CUDA}
#uv pip install torch==$TVERSION torchaudio==$TVERSION torchvision torchcodec  --index-url https://mirrors.nju.edu.cn/pytorch/whl/${BUILD_CUDA}   --index-strategy unsafe-best-match
uv pip install torch==$TORCH_VERSION torchaudio==$TORCH_VERSION torchvision torchcodec --index-url https://download.pytorch.org/whl/$BUILD_CUDA
uv pip install torch-complex "torch~=$TORCH_VERSION"
#uv pip install setuptools wheel ninja "torch~=$TORCH_VERSION"
uv pip install --upgrade pip setuptools wheel ninja "torch~=$TORCH_VERSION"
uv pip install matrix-nio "torch~=$TORCH_VERSION"
uv pip install sounddevice easydict pytorch_lightning "torch~=$TORCH_VERSION"
uv pip install silentcipher  --no-deps "torch~=$TORCH_VERSION"
uv pip install -r requirements.txt "torch~=$TORCH_VERSION"
uv pip install -r manager_requirements.txt "torch~=$TORCH_VERSION"
# 下面是插件的
#uv pip install opencv-python imageio-ffmpeg gguf scikit-image piexif segment_anything  "torch~=$TORCH_VERSION"
#uv pip install git+https://github.com/facebookresearch/sam2

rm -rf models
ln -s /data/ai/ai-model/comfyui/ ./models
rm -rf user/default
ln -s /data/ai-code/comfy-workflow ./user/default
rm -rf input
ln -s /data/ai-code/comfy-input ./input

uv pip show torch torchaudio torchvision flash-attn sageattention3


# 先卸载当前版本
pip uninstall kornia kornia_rs
# 安装不带 Rust 扩展的旧版本
pip install kornia==0.6.12  # 或更早版本
