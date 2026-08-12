#!/bin/bash
#

source .venv/bin/activate

git clone https://github.com/diodiogod/TTS-Audio-Suite.git custom_nodes/tts_audio_suite
uv pip install -r custom_nodes/tts_audio_suite/requirements.txt
