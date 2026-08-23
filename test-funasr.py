from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os

# ============ 配置区 ============
# 1. 指定模型下载存放位置
MODEL_DIR = "/data/ai-code/comfy-models/funasr"  # 根据你的实际路径调整
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. VAD 参数配置
VAD_MAX_SEGMENT_TIME = 30000  # 10秒 (单位: 毫秒)
VAD_MIN_SEGMENT_TIME = 300    # 最小语音片段 0.3秒
VAD_SILENCE_THRESHOLD = 0.3   # 静音检测阈值

# 3. 批处理参数
BATCH_SIZE_S = 60  # 动态批处理时长(秒)，可根据显存调整
# ================================

# 加载模型（指定下载目录）
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",
    #download_root=MODEL_DIR,  # 🔑 指定模型下载位置
    # VAD 参数控制
    vad_kwargs={
        "max_single_segment_time": VAD_MAX_SEGMENT_TIME,  # 最大分段时长 10秒
        "min_single_segment_time": VAD_MIN_SEGMENT_TIME,  # 最小分段时长
        "threshold": VAD_SILENCE_THRESHOLD,              # 静音检测阈值
    },
    # 模型缓存配置
    disable_download=False,  # 如果已下载，不会重复下载
    #cache_dir=MODEL_DIR,     # 缓存目录
)

# 处理音频文件
result = model.generate(
    input="/data/ai-code/comfy-input/头条开讲-5到25分钟.mp3",
    batch_size_s=BATCH_SIZE_S,  # 控制批处理大小
)
print(f"---------> {len(result)} {result[0].keys()}")
# 输出结果（带说话人ID和时间戳）
for seg in result[0]["sentence_info"]:
    start_time = seg['start'] / 1000  # 转换为秒
    end_time = seg.get('end', seg['start'] + 500) / 1000  # 结束时间（如果有）
    speaker = seg.get('spk', 'unknown')
    text = rich_transcription_postprocess(seg['sentence'])
    
    # 格式化输出
    print(f"[{start_time:.1f}s - {end_time:.1f}s] speaker_{speaker}: {text}")

# 可选：打印模型加载信息
print(f"\n✅ 模型已加载，存放位置: {MODEL_DIR}")
print(f"📊 VAD 最大分段: {VAD_MAX_SEGMENT_TIME/1000}秒")
