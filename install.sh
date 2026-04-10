#!/bin/bash

sudo apt install -y \
  libavdevice-dev \
  libavformat-dev \
  libavcodec-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  pkg-config

#安装 OpenMPI   tensorrt_llm 内部使用 mpi4py 来进行多 GPU 通信或分布式计算
sudo apt-get install libopenmpi-dev openmpi-bin
## --python=3.11 torch=2.8 有flash-attn=2.8
## --python=3.12 torch=2.9 有flash-attn=2.8
## uv venv --python 3.12 .venv
uv venv --python 3.12 .venv #3.10
export LD_PRELOAD=~/.local/share/uv/python/cpython-3.12.0-linux-x86_64-gnu/lib/libpython3.12.so.1.0
TORCH_VERSION=2.9.1 # $env:TORCH_VERSION=2.9.1 set TORCH_VERSION=2.9.1
BUILD_CUDA=cu130
mkdir -p deps

sed -i 's/VIRTUAL_ENV_PROMPT=.*/VIRTUAL_ENV_PROMPT="LTX-2"/g' .venv/bin/activate
source .venv/bin/activate

# wan2.2-animate使用
#uv pip install moviepy sam2 matplotlib onnxruntime

#uv pip install torch torchaudio torchvision "torch~=$TORCH_VERSION"  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130
#https://download.pytorch.org/whl/cu130 torchvision==0.24.0+${BUILD_CUDA}
uv pip install torch==${TORCH_VERSION}+${BUILD_CUDA} torchvision torchaudio torchcodec --index-url https://mirrors.nju.edu.cn/pytorch/whl/${BUILD_CUDA}   --index-strategy unsafe-best-match

uv pip install av tzdata pandas mpi4py "torch~=$TORCH_VERSION"

## 安装tensorrt-llm
export CFLAGS="-O2" CXXFLAGS="-O2"
#uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-cu13-libs tensorrt==10.12.0.36   "torch~=$TORCH_VERSION"
uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-cu13-libs tensorrt==10.15.1.29   "torch~=$TORCH_VERSION"
uv pip install llist==0.7.1 "torch~=$TORCH_VERSION"
uv pip install tensorrt-llm==1.3.0rc4 --extra-index-url https://pypi.nvidia.com  "torch~=$TORCH_VERSION" 

uv sync  --prerelease=allow
#uv pip install  onnx==1.15.0 #onnxruntime-gpu
uv pip install  onnx --only-binary :all: "torch~=$TORCH_VERSION"
 

## 安装 flash_attn2
uv pip install flash-attn --no-build-isolation "torch~=$TORCH_VERSION"
#uv pip install flashinfer-python  "torch~=$TORCH_VERSION"

## 安装sglang最新
#cd deps/sglang
#uv pip install -e "python"
#cd ../../

## 安装sglang
## uv pip install sglang "torch~=$TORCH_VERSION"

#uv pip install transformer_engine "torch~=$TORCH_VERSION"


## 安装flash_attn3
#git clone https://github.com/Dao-AILab/flash-attention.git .deps/flash-attention
#cd .deps/flash-attention/hopper
#CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
#cd ../../../


uv pip show torch torchaudio torchvision flash-attn

