#!/bin/bash
#

source .venv/bin/activate

git clone https://github.com/diodiogod/TTS-Audio-Suite.git custom_nodes/tts_audio_suite
uv pip install -r custom_nodes/tts_audio_suite/requirements.txt

git clone https://github.com/avenstack/ComfyUI-AV-FunASR.git custom_nodes/ComfyUI-AV-FunASR
uv pip install -r custom_nodes/ComfyUI-AV-FunASR/requirements.txt

uv pip install datasets==2.14.0
uv pip install huggingface-hub #==1.27.0
