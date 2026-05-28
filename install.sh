#!/bin/bash

sudo apt install -y \
  libavdevice-dev \
  libavformat-dev \
  libavcodec-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  ffmpeg \
  sox  portaudio19-dev \
  pkg-config

#安装 OpenMPI   tensorrt_llm 内部使用 mpi4py 来进行多 GPU 通信或分布式计算
sudo apt-get install libopenmpi-dev openmpi-bin
## --python=3.11 torch=2.8 有flash-attn=2.8
## --python=3.12 torch=2.9 有flash-attn=2.8
## uv venv --python 3.12 .venv
uv venv --python 3.12 .venv #3.10
#export LD_PRELOAD=~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/libpython3.12.so.1.0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0
TORCH_VERSION=2.11.0 # $env:TORCH_VERSION=2.9.1 set TORCH_VERSION=2.9.1
BUILD_CUDA=cu130
mkdir -p deps

sed -i 's/VIRTUAL_ENV_PROMPT=.*/VIRTUAL_ENV_PROMPT="comfyui"/g' .venv/bin/activate
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

## 安装 flash_attn2
uv pip install flash-attn --no-cache-dir --no-build-isolation "torch~=$TORCH_VERSION"

## 安装flash_attn3
#git clone https://github.com/Dao-AILab/flash-attention.git deps/flash-attention
#cd deps/flash-attention/hopper
#CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
#cd ../../../


git clone https://github.com/thu-ml/SageAttention.git deps/SageAttention
cd deps/SageAttention/sageattention3_blackwell
CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=32  python setup.py install

## 安装sglang最新
#cd deps/sglang
#uv pip install -e "python"
#cd ../../

## 安装sglang
## uv pip install sglang "torch~=$TORCH_VERSION"
#uv pip install transformer_engine "torch~=$TORCH_VERSION"





uv pip show torch torchaudio torchvision flash-attn sageattention3

