#!/bin/bash

mkdir -p deps

uv pip install build packaging==23 setuptools==74 wheel==0.43


## 安装 flash_attn2
#uv pip install flash-attn --no-cache-dir --no-build-isolation "torch~=$TORCH_VERSION"
## 安装flash_attn3
git clone https://github.com/Dao-AILab/flash-attention.git deps/flash-attention
cd deps/flash-attention
CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=16 python -m build --wheel --no-isolation
uv pip install dist/flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl
cd ../../
#cd deps/flash-attention/hopper
#CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
#CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=16 python -m build --wheel --no-isolation
#uv pip install dist/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
#cd ../../../


git clone https://github.com/thu-ml/SageAttention.git deps/SageAttention
#安装 sage2
cd deps/SageAttention

#CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=16  python setup.py install
CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=16 python -m build --wheel --no-isolation
uv pip install dist/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
cd ../../

#安装 sage3
cd deps/SageAttention/sageattention3_blackwell
#CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=32  python setup.py install
CFLAGS="-O2" CXXFLAGS="-O2" NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=16 python -m build --wheel --no-isolation
uv pip install dist/sageattn3-1.0.0-cp312-cp312-linux_x86_64.whl
cd ../../../

## 安装sglang最新
#cd deps/sglang
#uv pip install -e "python"
#cd ../../

## 安装sglang
## uv pip install sglang "torch~=$TORCH_VERSION"
#uv pip install transformer_engine "torch~=$TORCH_VERSION"


