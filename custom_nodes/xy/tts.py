import os
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ============ 模型配置（可按需调整） ============
MODEL_DIR = "/data/ai-code/comfy-models/funasr"          # 模型下载目录，请根据实际环境修改
VAD_MAX_SEGMENT_TIME = 30000                             # 最大分段时长 (毫秒)
VAD_MIN_SEGMENT_TIME = 300                               # 最小分段时长 (毫秒)
VAD_SILENCE_THRESHOLD = 0.3                              # 静音检测阈值
BATCH_SIZE_S = 60                                        # 动态批处理时长 (秒)
# =================================================

class SplitSpeaker:
    _model = None  # 类变量，复用模型实例

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"default": None, "placeholder": "说话人音频"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "text_speaker")
    FUNCTION = "split"
    CATEGORY = "xy/tts"

    def split(self, audio):
       return ("")


# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "XSplitSpeaker": SplitSpeaker,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "XSplitSpeaker": "X Split tts speaker",
}