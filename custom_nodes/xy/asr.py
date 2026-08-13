# -*- coding: utf-8 -*-
"""
ComfyUI XFunASR 节点
====================
- 模型下载到 ComfyUI/models/FunASR/
- 固定 vad_model = fsmn-vad，spk_model = cam++
- 支持 NVIDIA CUDA / AMD ROCm / 华为昇腾 NPU / CPU
- CPU offload 支持，任务完成后自动将模型转移到 CPU，释放 GPU 显存
- 支持 ModelScope 和 Hugging Face 双下载源
- 输出纯文本 + 每个说话人的句子列表 + 带时间戳的字幕

依赖安装：
    pip install funasr modelscope huggingface_hub torchaudio

注意：首次运行会自动下载模型，请保证网络畅通。
"""

import os
import tempfile
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import torch
import torchaudio
import folder_paths


# ----------------------------------------------------------------------
# 模型列表（可根据 FunASR README 自行扩充）
# ----------------------------------------------------------------------
MODEL_LIST = [
    {
        "name": "paraformer-zh",
        "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "description": "Paraformer 中文大模型",
    },
    {
        "name": "paraformer-en",
        "model_id": "iic/speech_paraformer-large_asr_nat-en-16k-common-vocab10020-pytorch",
        "description": "Paraformer 英文大模型",
    },
    {
        "name": "paraformer-zh-contextual",
        "model_id": "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "description": "Paraformer 中文热词模型",
    },
    {
        "name": "paraformer-zh-vad-punc",
        "model_id": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "description": "Paraformer 中文标点模型",
    },
    {
        "name": "Fun-ASR-MLT-Nano",
        "model_id": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "description": "Fun-ASR MLT Nano 多语言轻量级模型",
    },
    {
        "name": "SenseVoiceSmall",
        "model_id": "iic/SenseVoiceSmall",
        "description": "SenseVoice 小模型，支持多语言",
    },
    {
        "name": "Paraformer-zh-streaming",
        "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "description": "Paraformer 中文流式模型",
    },
    {
        "name": "Qwen3-ASR",
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "description": "Qwen3 ASR 1.7B 模型",
    },
]

# 模型根目录：ComfyUI/models/FunASR
MODELS_DIR = os.path.join(folder_paths.models_dir, "FunASR")
print(f"--- funasr model = {MODELS_DIR}")
os.makedirs(MODELS_DIR, exist_ok=True)


# 固定 vad / spk 模型
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" #"funasr/fsmn-vad" #
SPK_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common" #"funasr/campplus"

# 2. VAD 参数配置
VAD_MAX_SEGMENT_TIME = 30000  # 10秒 (单位: 毫秒)
VAD_MIN_SEGMENT_TIME = 300    # 最小语音片段 0.3秒
VAD_SILENCE_THRESHOLD = 0.3   # 静音检测阈值



# ----------------------------------------------------------------------
# 模型缓存管理
# ----------------------------------------------------------------------
@dataclass
class ModelCacheEntry:
    """模型缓存条目"""
    model: Any
    device: str  # 当前模型所在的设备
    last_used: float  # 最后使用时间戳
    

class ModelCacheManager:
    """管理模型缓存，支持 CPU offload"""
    
    def __init__(self, max_cached_models: int = 3):
        self.cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self.max_cached_models = max_cached_models
        
    def get(self, key: str) -> Optional[ModelCacheEntry]:
        """获取缓存的模型"""
        if key in self.cache:
            # 移到末尾（最近使用）
            entry = self.cache.pop(key)
            self.cache[key] = entry
            return entry
        return None
    
    def put(self, key: str, entry: ModelCacheEntry):
        """添加模型到缓存"""
        if key in self.cache:
            self.cache.pop(key)
        
        self.cache[key] = entry
        
        # 如果缓存过多，移除最旧的
        while len(self.cache) > self.max_cached_models:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self._cleanup_entry(oldest_key, oldest_entry)
    
    def _cleanup_entry(self, key: str, entry: ModelCacheEntry):
        """清理缓存条目，释放内存"""
        print(f"[XFunASR] 清理缓存模型: {key}")
        try:
            if hasattr(entry.model, 'model') and entry.model.model is not None:
                del entry.model.model
            if hasattr(entry.model, 'vad_model') and entry.model.vad_model is not None:
                del entry.model.vad_model
            if hasattr(entry.model, 'spk_model') and entry.model.spk_model is not None:
                del entry.model.spk_model
            del entry.model
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
    
    def clear_all(self):
        """清理所有缓存"""
        for key, entry in list(self.cache.items()):
            self._cleanup_entry(key, entry)
        self.cache.clear()


# 全局模型缓存管理器
_MODEL_CACHE_MANAGER = ModelCacheManager(max_cached_models=3)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def get_local_model_dir(repo_id: str) -> str:
    """将 repo_id 映射到本地 ComfyUI/models/FunASR/ 下的目录"""
    name = repo_id.split("/")[-1]
    return os.path.join(MODELS_DIR, name)


def download_model(repo_id: str) -> str:
    """下载模型到 ComfyUI/models/FunASR/ 并返回本地路径
    
    优先从 ModelScope 下载，如果失败则从 Hugging Face 下载
    """
    local_dir = get_local_model_dir(repo_id)

    # 如果目录已经存在且非空，则直接返回，避免重复下载
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[XFunASR] 模型已存在: {local_dir}")
        return local_dir

    print(f"[XFunASR] 正在下载模型 {repo_id} 到 {local_dir} ...")
    
    # 1. 尝试从 ModelScope 下载
    try:
        from modelscope import snapshot_download
        print(f"[XFunASR] 尝试从 ModelScope 下载: {repo_id}")
        snapshot_download(repo_id, local_dir=local_dir)
        print(f"[XFunASR] ModelScope 下载完成：{local_dir}")
        return local_dir
    except Exception as e:
        print(f"[XFunASR] ModelScope 下载失败: {e}")
        print(f"[XFunASR] 尝试从 Hugging Face 下载...")
    
    # 2. 尝试从 Hugging Face 下载
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        
        # 转换 repo_id 格式（通常相同，但可能需要处理特殊情况）
        hf_repo_id = repo_id
        
        print(f"[XFunASR] 从 Hugging Face 下载: {hf_repo_id}")
        
        # 尝试下载
        hf_snapshot_download(
            repo_id=hf_repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"[XFunASR] Hugging Face 下载完成：{local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"[XFunASR] Hugging Face 下载失败: {e}")
        
        # 清理可能的不完整下载
        if os.path.isdir(local_dir) and not os.listdir(local_dir):
            try:
                os.rmdir(local_dir)
            except:
                pass
        
        raise RuntimeError(
            f"模型下载失败: {repo_id}\n"
            f"ModelScope 错误: 请检查模型 ID 是否正确\n"
            f"Hugging Face 错误: {e}\n"
            f"请手动下载模型到: {local_dir}"
        )


def resolve_device(device_choice: str) -> str:
    """
    将 UI 选项映射为 PyTorch 设备字符串
    
    参数:
        device_choice: 'cuda' (NVIDIA CUDA), 'rocm' (AMD ROCm), 'npu' (华为昇腾), 'cpu' (CPU)
    
    返回:
        PyTorch 设备字符串: 'cuda:0', 'npu:0', 'cpu'
    """
    if device_choice == "cuda":
        # NVIDIA CUDA
        if torch.cuda.is_available():
            return "cuda:0"
        raise RuntimeError("未检测到 NVIDIA CUDA GPU，请确认已安装 CUDA 版本的 PyTorch")

    if device_choice == "rocm":
        # AMD ROCm - PyTorch ROCm 版本同样通过 cuda 接口访问
        if torch.cuda.is_available():
            # 可以额外检查是否是 ROCm 版本
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return "cuda:0"
            else:
                print("[XFunASR] 警告: 当前 PyTorch 不是 ROCm 版本，但仍尝试使用 CUDA 接口")
                return "cuda:0"
        raise RuntimeError("未检测到 AMD ROCm GPU，请确认已安装 ROCm 版本的 PyTorch")

    if device_choice == "npu":
        # 华为昇腾 NPU
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            raise RuntimeError("使用昇腾 NPU 需要安装 torch_npu: pip install torch-npu")
        
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
        raise RuntimeError("未检测到华为昇腾 NPU，请确认已正确安装 torch_npu")

    if device_choice == "cpu":
        # CPU
        return "cpu"

    raise ValueError(f"未知设备选项: {device_choice}，支持: cuda, rocm, npu, cpu")


def move_model_to_device(model, device: str):
    """将 FunASR 模型移动到指定设备"""
    try:
        # 移动主模型
        if hasattr(model, 'model') and model.model is not None:
            model.model = model.model.to(device)
        
        # 移动 VAD 模型
        if hasattr(model, 'vad_model') and model.vad_model is not None:
            if hasattr(model.vad_model, 'model'):
                model.vad_model.model = model.vad_model.model.to(device)
            else:
                model.vad_model = model.vad_model.to(device)
        
        # 移动说话人识别模型
        if hasattr(model, 'spk_model') and model.spk_model is not None:
            if hasattr(model.spk_model, 'model'):
                model.spk_model.model = model.spk_model.model.to(device)
            else:
                model.spk_model = model.spk_model.to(device)
        
        # 更新模型的 device 属性
        if hasattr(model, 'device'):
            model.device = device
        
        print(f"[XFunASR] 模型已移动到 {device}")
        return True
    except Exception as e:
        print(f"[XFunASR] 移动模型到 {device} 失败: {e}")
        return False


def offload_model_to_cpu(model):
    """将 FunASR 模型 offload 到 CPU，释放 GPU 显存"""
    if not move_model_to_device(model, "cpu"):
        print("[XFunASR] 警告: CPU offload 失败")
        return False
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
    
    print("[XFunASR] 模型已 offload 到 CPU，GPU 显存已释放")
    return True


def parse_speaker_map(speaker_map: str) -> dict:
    """解析用户输入：'0:lilei, 1:liming' -> {'0': 'lilei', '1': 'liming'}"""
    mapping = {}
    if not speaker_map:
        return mapping

    speaker_map = speaker_map.replace("：", ":")
    for part in speaker_map.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
            mapping[k.strip()] = v.strip()

    return mapping


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def audio_to_temp_wav(audio: dict) -> str:
    """将 ComfyUI AUDIO 对象转换为 16k 单声道 wav 临时文件"""
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate", 16000)

    if waveform is None:
        raise RuntimeError("输入 audio 中缺少 waveform 字段")

    waveform = waveform.detach().cpu()

    # 处理不同维度的波形数据
    print(f"[XFunASR] 原始 waveform 维度: {waveform.shape}")
    
    if waveform.dim() == 1:
        # [samples] -> [1, samples]
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        # 判断是 [channels, samples] 还是 [samples, channels]
        # ComfyUI 通常是 [channels, samples]，但有时可能是 [samples, channels]
        if waveform.shape[0] <= 2 and waveform.shape[1] > 2:
            # [channels, samples]，保持不变
            pass
        elif waveform.shape[1] <= 2 and waveform.shape[0] > 2:
            # [samples, channels]，转置为 [channels, samples]
            waveform = waveform.transpose(0, 1)
        else:
            # 无法判断，假设是 [channels, samples]
            pass
    elif waveform.dim() == 3:
        # [batch, channels, samples] -> [channels, samples]
        # 取第一个 batch
        waveform = waveform.squeeze(0)
        if waveform.dim() == 3:
            # 如果 squeeze 后还是 3 维，取第一个
            waveform = waveform[0]
        elif waveform.dim() == 1:
            # 如果是 [1, 2, samples] -> squeeze -> [2, samples]
            pass
    else:
        raise RuntimeError(f"不支持的 audio waveform 维度: {waveform.shape}")

    # 确保是 2D: [channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 3:
        # 如果还是 3 维，取第一个 batch 和第一个 channel
        waveform = waveform[0, 0:1, :]
    
    print(f"[XFunASR] 处理后 waveform 维度: {waveform.shape}")

    # 转换为 float32
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
        if waveform.abs().max() > 1.5:
            waveform = waveform / 32768.0
    else:
        if waveform.abs().max() > 1.5:
            waveform = waveform / 32768.0

    # 转为单声道（如果有多声道）
    if waveform.shape[0] > 1:
        mono = waveform.mean(dim=0, keepdim=True)  # [1, samples]
    else:
        mono = waveform  # [1, samples]

    # 确保是 [1, samples]
    if mono.dim() == 1:
        mono = mono.unsqueeze(0)

    # 重采样到 16k
    if sample_rate != 16000:
        mono = torchaudio.functional.resample(mono, sample_rate, 16000)

    # 写入临时 wav
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(tmp_path, mono, 16000)

    return tmp_path

def rich_transcription_postprocess(text: str) -> str:
    """处理富文本转录结果，提取纯文本和说话人信息
    
    处理 SenseVoice 等模型的富文本输出格式
    例如: <|zh|><|NEUTRAL|><|Speech|><|woitn|>你好，今天天气不错
    
    参数:
        text: 原始转录文本，可能包含富文本标签
        
    返回:
        清理后的纯文本
    """
    if not text:
        return ""
    
    # 移除语言标签 <|zh|>, <|en|>, <|ja|> 等
    import re
    text = re.sub(r'<\|[a-z]{2,3}\|>', '', text)
    
    # 移除情感标签 <|NEUTRAL|>, <|HAPPY|>, <|SAD|> 等
    text = re.sub(r'<\|[A-Z]+\|>', '', text)
    
    # 移除事件标签 <|Speech|>, <|Music|>, <|Laughter|> 等
    text = re.sub(r'<\|[A-Za-z]+\|>', '', text)
    
    # 移除说话人标签 <|spk[0-9]|> 等
    text = re.sub(r'<\|spk\d+\|>', '', text)
    
    # 移除时间戳标签 <|0.00|>, <|1.50|> 等
    text = re.sub(r'<\|\d+\.\d+\|>', '', text)
    
    # 移除其他特殊标签 <|xxx|>
    text = re.sub(r'<\|[^|]+\|>', '', text)
    
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ----------------------------------------------------------------------
# ComfyUI 节点
# ----------------------------------------------------------------------
class XFunASR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    [m["name"] for m in MODEL_LIST],
                    {"default": MODEL_LIST[0]["name"]},
                ),
                "device": (
                    ["cuda", "rocm", "npu", "cpu"],  # 直接显示技术框架名称
                    {"default": "cuda"},
                ),
                "audio": ("AUDIO",),
                "speaker_map": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "0:lilei, 1:liming",
                    },
                ),
                "enable_cpu_offload": (
                    "BOOLEAN",
                    {"default": True},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "DICT",)
    RETURN_NAMES = ("text", "sentence", "srt", "map")
    FUNCTION = "run"
    CATEGORY = "xy/audio"

    def _get_or_load_model(self, model_id: str, device: str, enable_cpu_offload: bool):
        """获取或加载模型，支持 CPU offload"""
        
        cache_key = f"{model_id}_{device}"
        cache_entry = _MODEL_CACHE_MANAGER.get(cache_key)
        
        if cache_entry is not None:
            # 模型在缓存中
            model = cache_entry.model
            current_device = cache_entry.device
            
            # 如果目标设备不是 CPU，且模型当前在 CPU 上，需要移回 GPU
            if device != "cpu" and current_device == "cpu":
                print(f"[XFunASR] 从 CPU 移回模型到 {device} ...")
                if not move_model_to_device(model, device):
                    # 移动失败，重新加载
                    cache_entry = None
                else:
                    cache_entry.device = device
                    cache_entry.last_used = time.time()
            elif device == "cpu" and current_device != "cpu" and enable_cpu_offload:
                # 已经在 CPU 上
                cache_entry.last_used = time.time()
            else:
                cache_entry.last_used = time.time()
        
        if cache_entry is None:
            # 需要加载新模型
            try:
                from funasr import AutoModel
            except ImportError:
                raise RuntimeError(
                    "未安装 funasr，请先执行: pip install funasr"
                )
            
            local_model_dir = download_model(model_id)
            local_vad_dir = download_model(VAD_MODEL_ID)
            local_spk_dir = download_model(SPK_MODEL_ID)
            
            print(f"[XFunASR] 正在加载模型到 {device} ...")
            print(f"[XFunASR] model {local_model_dir}")
            print(f"[XFunASR] vad_model {local_vad_dir}")
            print(f"[XFunASR] spk_model {local_spk_dir}")
            # 加载模型
            model = AutoModel(
                model=local_model_dir,
                vad_model=local_vad_dir,
                spk_model=local_spk_dir,
                device=device,
                disable_update=True,
                disable_download=False,  # 如果已下载，不会重复下载
                log_level="ERROR",
                # VAD 参数控制
                vad_kwargs={
                    "max_single_segment_time": VAD_MAX_SEGMENT_TIME,  # 最大分段时长 10秒
                    "min_single_segment_time": VAD_MIN_SEGMENT_TIME,  # 最小分段时长
                    "threshold": VAD_SILENCE_THRESHOLD,              # 静音检测阈值
                },

            )
            
            cache_entry = ModelCacheEntry(
                model=model,
                device=device,
                last_used=time.time()
            )
            _MODEL_CACHE_MANAGER.put(cache_key, cache_entry)
            
            print(f"[XFunASR] 模型加载完成: {cache_key}")
        
        return cache_entry.model

    def run(self, model, device, audio, speaker_map="", enable_cpu_offload=True):
        # 1. 解析设备
        device_str = resolve_device(device)
        
        # 2. 获取模型
        model_meta = next(m for m in MODEL_LIST if m["name"] == model)
        model_id = model_meta["model_id"]
        
        asr_model = self._get_or_load_model(model_id, device_str, enable_cpu_offload)
        
        # 3. 音频转临时 wav
        tmp_wav = audio_to_temp_wav(audio)
        
        try:
            # 4. 识别
            print(f"[XFunASR] 开始识别音频...")
            res = asr_model.generate(
                input=tmp_wav,
                batch_size_s=60,
                #sentence_timestamp=True,
            )
            print(f"[XFunASR] 识别完成 {len(res)}")
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            
            # 5. CPU offload：任务完成后将模型移到 CPU，释放 GPU 显存
            if enable_cpu_offload and device_str != "cpu":
                # 更新缓存中的设备信息
                cache_key = f"{model_id}_{device_str}"
                cache_entry = _MODEL_CACHE_MANAGER.get(cache_key)
                if cache_entry is not None:
                    if offload_model_to_cpu(asr_model):
                        cache_entry.device = "cpu"
                        cache_entry.last_used = time.time()
        
        # 6. 应用说话人名称映射
        mapping = parse_speaker_map(speaker_map)
        print(f"--->speaker_map={speaker_map} mapping={mapping}")
        # 7. 解析结果

        all_texts = []
        speaker_sentences = {}
        subtitle_lines = []
        subtitle_index = 1

        info = res[0]
        # ("text", "text_cn", "text", "srt", "map")
        # 输出结果
        sentences = []
        srts = []
        speaker_map = {}
        #for seg in info["sentence_info"]:
        idx = 0
        for seg in info["sentence_info"]:
            start_time = seg['start'] / 1000  # 转换为秒
            end_time = seg.get('end', seg['start'] + 500) / 1000  # 结束时间（如果有）
            spk = f"{seg.get('spk', 'unknown')}"
            speaker = mapping.get(spk, spk) if spk in mapping else f"speaker_{spk}"
            text = rich_transcription_postprocess(seg['sentence'])

            if len(text) < 1 :
                continue
            idx = idx + 1

            sentences.append(f"[{speaker}] {text}")

            srts.append(f"{idx}")
            srts.append(f"{start_time:.1f} --> {end_time:.1f}")
            srts.append(f"{text}")

            if speaker in speaker_map:
                speaker_map[speaker].extend(text)
            else:
                speaker_map[speaker] = list(text)

            # 格式化输出
            #print(f"[{start_time:.1f}s - {end_time:.1f}s] {speaker}: {text}")
        
        sentence = "\n".join(sentences)
        srt_text = "\n".join(srts)
        #print(f"keys = {info.keys()} {info['key']} {info['label']}")
        return (info["text"], sentence, srt_text, speaker_map)


# ----------------------------------------------------------------------
# ComfyUI 节点注册
# ----------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "XFunASR": XFunASR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFunASR": "XFunASR",
}