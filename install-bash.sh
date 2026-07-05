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

mkdir -p /data/ai-code/.cache/uv