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
                "speaker_map": ("STRING", {
                    "default": "",
                    "placeholder": "例如: 0:luo,1:li",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "text_speaker")
    FUNCTION = "split"
    CATEGORY = "xy/tts"

    @classmethod
    def load_model(cls):
        """加载或获取缓存的模型实例"""
        if cls._model is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            cls._model = AutoModel(
                model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
                vad_model="fsmn-vad",
                spk_model="cam++",
                device="cuda",
                vad_kwargs={
                    "max_single_segment_time": VAD_MAX_SEGMENT_TIME,
                    "min_single_segment_time": VAD_MIN_SEGMENT_TIME,
                    "threshold": VAD_SILENCE_THRESHOLD,
                },
                disable_download=False,
                # cache_dir=MODEL_DIR,  # 根据 funasr 版本可能使用 download_root
                download_root=MODEL_DIR,  # 指定模型下载位置
            )
        return cls._model

    def split(self, audio, speaker_map):
        """
        核心处理方法
        :param audio: 音频文件路径 (字符串) 或可能的音频数据（暂按路径处理）
        :param speaker_map: "0:luo,1:li" 格式的字符串
        :return: (纯文本, 带说话人标签的文本)
        """
        # 1. 解析 speaker_map
        speaker_dict = {}
        if speaker_map and speaker_map.strip():
            for item in speaker_map.split(','):
                item = item.strip()
                if ':' in item:
                    k, v = item.split(':', 1)
                    speaker_dict[k.strip()] = v.strip()

        # 2. 加载模型
        model = self.load_model()

        # 3. 执行推理
        # 注意：audio 可能是文件路径字符串，也可能是其他格式，这里假设为路径
        if not isinstance(audio, str):
            # 如果传入的是 ComfyUI 音频对象（如包含路径的属性），尝试提取
            # 但为了兼容，可尝试将 audio 转为字符串路径
            audio = str(audio) if hasattr(audio, "__str__") else audio
        result = model.generate(
            input=audio,
            batch_size_s=BATCH_SIZE_S,
        )

        # 4. 处理结果，构建文本
        if not result or len(result) == 0:
            return ("", "")

        segments = result[0].get("sentence_info", [])
        if not segments:
            return ("", "")

        plain_text_parts = []
        speaker_text_parts = []

        for seg in segments:
            # 获取说话人 ID 并映射为名称
            spk_id = str(seg.get('spk', 'unknown'))
            speaker_name = speaker_dict.get(spk_id, f"speaker_{spk_id}")  # 若未映射则保留原始ID

            # 获取文本并后处理（去除多余空格等）
            raw_text = seg.get('sentence', '')
            text = rich_transcription_postprocess(raw_text)

            # 收集纯文本（按顺序拼接）
            plain_text_parts.append(text)

            # 收集带说话人标签的文本
            speaker_text_parts.append(f"[{speaker_name}] {text}")

        # 拼接最终结果
        plain_text = "".join(plain_text_parts)          # 纯文本直接连接（可根据需要加分隔符）
        speaker_text = "\n".join(speaker_text_parts)    # 每句一行，方便阅读

        return (plain_text, speaker_text)


# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "XSplitSpeaker": SplitSpeaker,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "XSplitSpeaker": "X Split tts speaker",
}