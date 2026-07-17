import os
import re
import subprocess
import tempfile
import json
from pathlib import Path
import folder_paths

# 尝试导入 ComfyUI 的进度工具
try:
    from comfy.utils import ProgressBar
    HAS_COMFY_PROGRESS = True
except ImportError:
    HAS_COMFY_PROGRESS = False
    print("[XVideoMergeNode] ComfyUI progress bar not available")

# 尝试导入 imageio-ffmpeg
try:
    import imageio_ffmpeg as ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False
    print("[XVideoMergeNode] imageio-ffmpeg not available, please install: pip install imageio-ffmpeg")


def natural_sort_key(filename):
    """
    自然排序：提取文件名中的数字进行排序
    例如: video-1-audio.mp4, video-2-audio.mp4, video-10-audio.mp4
    """
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[0])
    return 0


class ProgressTracker:
    """进度追踪器，用于在 ComfyUI 中显示进度"""
    
    def __init__(self, total_steps, node_instance=None):
        self.total_steps = total_steps
        self.current_step = 0
        self.node_instance = node_instance
        self.pbar = None
        
        if HAS_COMFY_PROGRESS and total_steps > 0:
            self.pbar = ProgressBar(total_steps)
    
    def update(self, step=None, advance=None, description=""):
        """更新进度"""
        if step is not None:
            self.current_step = step
        elif advance is not None:
            self.current_step += advance
        else:
            self.current_step += 1
        
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        
        if self.pbar is not None:
            self.pbar.update(self.current_step)
        
        if self.node_instance is not None:
            try:
                if hasattr(self.node_instance, 'set_title'):
                    self.node_instance.set_title(f"合并视频 {self.current_step}/{self.total_steps}")
            except:
                pass
        
        print(f"[XVideoMergeNode] 进度: {self.current_step}/{self.total_steps} - {description}")
    
    def close(self):
        """关闭进度条"""
        self.pbar = None


def get_ffmpeg_path():
    """获取 ffmpeg 可执行文件路径"""
    if HAS_FFMPEG:
        try:
            return ffmpeg.get_ffmpeg_exe()
        except:
            pass
    
    # 尝试从系统路径查找
    for cmd in ['ffmpeg', 'ffmpeg.exe']:
        try:
            result = subprocess.run([cmd, '-version'], capture_output=True, timeout=1)
            if result.returncode == 0:
                return cmd
        except:
            pass
    
    raise RuntimeError("ffmpeg not found. Please install ffmpeg or imageio-ffmpeg")


def validate_audio_input(audio_clip):
    """
    验证并解析音频输入
    
    Args:
        audio_clip: ComfyUI 的音频对象 (AUDIO 类型)
    
    Returns:
        tuple: (waveform, sample_rate, is_valid)
    """
    if audio_clip is None:
        return None, None, False
    
    import torch
    import numpy as np
    
    # 如果是字典，提取 waveform 和 sample_rate
    if isinstance(audio_clip, dict):
        waveform = audio_clip.get('waveform')
        sample_rate = audio_clip.get('sample_rate', 44100)
        
        if waveform is None:
            raise ValueError("音频字典中缺少 'waveform' 键")
    else:
        # 如果是对象，尝试直接访问属性
        if hasattr(audio_clip, 'waveform'):
            waveform = audio_clip.waveform
            sample_rate = getattr(audio_clip, 'sample_rate', 44100)
        else:
            raise ValueError("无法解析音频输入，请确保提供有效的 AUDIO 类型数据")
    
    # 转换为 numpy 数组
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()
    
    print(f"[XVideoMergeNode] 原始音频形状: {waveform.shape}")
    
    # 处理不同形状的音频数据
    # 常见格式:
    # - (samples,): 单声道
    # - (samples, channels): 多声道
    # - (channels, samples): 多声道
    # - (1, channels, samples): ComfyUI 可能返回的格式
    
    if waveform.ndim == 3:
        # 如果是 (1, channels, samples) 格式，去掉第一维
        if waveform.shape[0] == 1:
            waveform = waveform[0]  # 变为 (channels, samples)
        else:
            # 其他3维格式，尝试合并
            waveform = waveform.reshape(-1, waveform.shape[-1])
    
    # 确保是2维数组
    if waveform.ndim == 1:
        # 单声道: (samples,) -> (1, samples)
        waveform = waveform.reshape(1, -1)
    elif waveform.ndim == 2:
        # 如果是 (samples, channels) 格式，转置为 (channels, samples)
        # 判断依据：如果第二维较小（通常 <= 8），可能是通道数
        if waveform.shape[1] <= 8 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T
    
    # 确保形状为 (channels, samples)
    if waveform.ndim != 2:
        raise ValueError(f"无法解析音频形状: {waveform.shape}")
    
    # 归一化处理
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype == np.uint8:
        waveform = (waveform.astype(np.float32) - 128) / 128.0
    
    channels, samples = waveform.shape
    audio_duration = samples / sample_rate
    
    print(f"[XVideoMergeNode] 音频解析成功 - 声道数: {channels}, 采样数: {samples}, 采样率: {sample_rate}, 时长: {audio_duration:.2f}秒")
    
    return waveform, sample_rate, True


def create_audio_file(waveform, sample_rate, output_path):
    """
    使用 ffmpeg 将音频数据写入文件
    
    Args:
        waveform: numpy 数组格式的音频数据 (channels, samples)
        sample_rate: 采样率
        output_path: 输出文件路径
    """
    import numpy as np
    import subprocess
    import tempfile
    import os
    
    # 确保数据格式正确
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    
    # 确保形状为 (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    
    # 写入临时 WAV 文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    try:
        # 转换为 int16
        audio_int16 = (waveform * 32767).astype(np.int16)
        
        # 使用 scipy 写入 wav（如果没有，使用纯 Python 实现）
        try:
            from scipy.io import wavfile
            # scipy 期望 (samples, channels)
            wavfile.write(temp_wav_path, sample_rate, audio_int16.T)
        except ImportError:
            # 使用 wave 模块
            import wave
            channels = audio_int16.shape[0]
            with wave.open(temp_wav_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.T.tobytes())
        
        # 使用 ffmpeg 转换格式
        subprocess.run([
            get_ffmpeg_path(),
            '-i', temp_wav_path,
            '-c:a', 'aac',
            '-b:a', '192k',
            '-y',
            output_path
        ], check=True, capture_output=True)
        
        print(f"[XVideoMergeNode] 音频文件已创建: {output_path}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def process_video_file(video_path, audio_path, output_path, progress_callback=None):
    """
    使用 ffmpeg 处理单个视频文件（合并音频）
    
    Args:
        video_path: 输入视频路径
        audio_path: 输入音频路径（可选）
        output_path: 输出视频路径
        progress_callback: 进度回调函数
    """
    ffmpeg_cmd = get_ffmpeg_path()
    
    # 构建 ffmpeg 命令
    cmd = [ffmpeg_cmd, '-i', video_path]
    
    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-i', audio_path])
        # 使用音频替换视频中的音频
        cmd.extend(['-c:v', 'copy'])  # 视频流直接复制
        cmd.extend(['-map', '0:v:0'])  # 使用第一个输入的视频
        cmd.extend(['-map', '1:a:0'])  # 使用第二个输入的音频
        cmd.extend(['-shortest'])  # 以最短的流为准
    else:
        # 只复制视频流，移除音频
        cmd.extend(['-an'])  # 移除音频
        cmd.extend(['-c:v', 'copy'])
    
    cmd.extend(['-y', output_path])
    
    # 执行 ffmpeg 命令
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 处理视频失败: {e.stderr}")


def merge_videos(video_files, output_path, progress_callback=None):
    """
    合并多个视频文件
    
    Args:
        video_files: 视频文件列表
        output_path: 输出文件路径
        progress_callback: 进度回调函数
    """
    if not video_files:
        raise ValueError("视频文件列表为空")
    
    # 创建临时文件列表
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_file in video_files:
            f.write(f"file '{os.path.abspath(video_file)}'\n")
        concat_file = f.name
    
    try:
        ffmpeg_cmd = get_ffmpeg_path()
        
        # 使用 concat demuxer 合并视频
        cmd = [
            ffmpeg_cmd,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # 不重新编码
            '-y',
            output_path
        ]
        
        # 执行合并
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"合并视频失败: {e.stderr}")
    finally:
        # 清理临时文件
        if os.path.exists(concat_file):
            os.remove(concat_file)


def get_video_duration(video_path):
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        float: 视频时长（秒）
    """
    ffmpeg_cmd = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_cmd,
        '-i', video_path,
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        # 从输出中提取时长
        import re
        match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)', result.stderr)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            return hours * 3600 + minutes * 60 + seconds
        return 0
    except:
        return 0


def merge_videos_with_audio(video_dir, audio_clip, output_file, progress_callback=None):
    """
    合并指定目录下所有 *-audio.mp4 文件，并替换音频
    
    Args:
        video_dir: 视频文件所在目录
        audio_clip: ComfyUI 的音频对象 (AUDIO 类型)，可以为 None
        output_file: 输出文件名
        progress_callback: 进度回调函数
    
    Returns:
        str: 输出文件路径
    """
    # ===== 1. 路径规范化 =====
    if not os.path.isabs(video_dir):
        video_dir = os.path.abspath(video_dir)
    
    if output_file and not os.path.isabs(output_file):
        output_file = os.path.join(video_dir, output_file)
    elif output_file:
        output_file = os.path.abspath(output_file)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # ===== 2. 验证输入 =====
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")
    
    # ===== 3. 收集视频文件 =====
    video_files = collect_video_files(video_dir)
    if not video_files:
        raise ValueError(f"在 {video_dir} 中未找到任何 *-audio.mp4 文件")
    
    # ===== 4. 计算总步骤 =====
    has_audio = audio_clip is not None
    total_steps = len(video_files) + 3 + (2 if has_audio else 0)
    tracker = ProgressTracker(total_steps, progress_callback)
    
    try:
        # ===== 5. 处理音频 =====
        audio_file = None
        if has_audio:
            tracker.update(description="处理音频输入")
            audio_file = process_audio_input(audio_clip, video_files, tracker)
        
        # ===== 6. 处理每个视频 =====
        processed_videos = process_all_videos(video_files, audio_file, tracker)
        
        # ===== 7. 合并视频 =====
        tracker.update(description="合并视频文件")
        merge_videos(processed_videos, output_file, tracker.update)
        
        # ===== 8. 清理临时文件 =====
        cleanup_temp_files(processed_videos, audio_file)
        
        # ===== 9. 完成 =====
        tracker.update(description="完成 ✅")
        tracker.close()
        
        print(f"[XVideoMergeNode] ✅ 合并完成！输出文件: {output_file}")
        return output_file
        
    except Exception as e:
        tracker.close()
        raise RuntimeError(f"视频合并失败: {str(e)}")


def collect_video_files(video_dir):
    """
    收集目录中的所有视频文件并按自然排序
    
    Args:
        video_dir: 视频目录
    
    Returns:
        list: 排序后的视频文件路径列表
    """
    video_files = []
    for filename in os.listdir(video_dir):
        if filename.endswith('-audio.mp4'):
            full_path = os.path.join(video_dir, filename)
            if os.path.isfile(full_path):
                video_files.append(full_path)
    
    video_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    print(f"[XVideoMergeNode] 找到 {len(video_files)} 个视频文件:")
    for f in video_files:
        print(f"  - {os.path.basename(f)}")
    
    return video_files


def process_audio_input(audio_clip, video_files, tracker):
    """
    处理音频输入，生成音频文件
    
    Args:
        audio_clip: ComfyUI 音频对象
        video_files: 视频文件列表
        tracker: 进度追踪器
    
    Returns:
        str: 音频文件路径
    """
    # 验证音频输入
    waveform, sample_rate, is_valid = validate_audio_input(audio_clip)
    if not is_valid:
        return None
    
    # 获取总视频时长
    tracker.update(description="计算视频时长")
    total_duration = sum(get_video_duration(vf) for vf in video_files)
    print(f"[XVideoMergeNode] 总视频时长: {total_duration:.2f}秒")
    
    # 获取音频时长
    channels, samples = waveform.shape
    audio_duration = samples / sample_rate
    print(f"[XVideoMergeNode] 音频原始时长: {audio_duration:.2f}秒")
    
    # 由于用户说音频长度一定 >= 视频总长度，直接截断
    tracker.update(description="调整音频长度")
    target_samples = int(total_duration * sample_rate)
    
    if samples > target_samples:
        print(f"[XVideoMergeNode] 音频长度 {audio_duration:.2f}s >= 目标长度 {total_duration:.2f}s，截断音频")
        waveform_adjusted = waveform[:, :target_samples]
    else:
        print(f"[XVideoMergeNode] 音频长度 {audio_duration:.2f}s < 目标长度 {total_duration:.2f}s，需要循环填充")
        # 如果音频比目标短（虽然用户说不会发生，但还是做个保护）
        waveform_adjusted = loop_audio_safe(waveform, target_samples, sample_rate)
    
    # 创建音频临时文件
    tracker.update(description="创建音频文件")
    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as f:
        audio_path = f.name
    
    # 写入音频文件
    create_audio_file(waveform_adjusted, sample_rate, audio_path)
    
    return audio_path


def loop_audio_safe(waveform, target_samples, sample_rate):
    """
    安全地循环填充音频（避免内存溢出）
    
    Args:
        waveform: 音频波形数据 (channels, samples)
        target_samples: 目标采样数
        sample_rate: 采样率
    
    Returns:
        numpy.ndarray: 循环填充后的音频
    """
    import numpy as np
    
    channels, current_samples = waveform.shape
    
    # 如果目标采样数小于等于当前采样数，直接截断
    if target_samples <= current_samples:
        return waveform[:, :target_samples]
    
    print(f"[XVideoMergeNode] 需要将音频从 {current_samples} 采样循环填充到 {target_samples} 采样")
    
    # 计算需要循环的次数
    repeats = (target_samples // current_samples) + 1
    print(f"[XVideoMergeNode] 需要循环 {repeats} 次")
    
    # 预分配结果数组
    result = np.zeros((channels, target_samples), dtype=waveform.dtype)
    
    # 逐声道处理，避免内存爆炸
    for ch in range(channels):
        # 使用 np.tile 但只处理单声道
        channel_data = waveform[ch, :]
        # 由于只处理单声道，内存占用很小
        repeated = np.tile(channel_data, repeats)
        result[ch, :] = repeated[:target_samples]
    
    print(f"[XVideoMergeNode] 循环填充完成，结果形状: {result.shape}")
    return result


def adjust_audio_length(waveform, sample_rate, target_duration):
    """
    调整音频长度以匹配目标时长（已弃用，改用 process_audio_input 中的直接截断）
    
    Args:
        waveform: 音频波形数据 (channels, samples)
        sample_rate: 采样率
        target_duration: 目标时长（秒）
    
    Returns:
        numpy.ndarray: 调整后的音频数据
    """
    import numpy as np
    
    channels, current_samples = waveform.shape
    target_samples = int(target_duration * sample_rate)
    current_duration = current_samples / sample_rate
    
    print(f"[XVideoMergeNode] 音频原始时长: {current_duration:.2f}s, 目标时长: {target_duration:.2f}s")
    
    # 如果音频时长 >= 目标时长，直接截断
    if current_samples >= target_samples:
        print(f"[XVideoMergeNode] 音频长度 >= 目标长度，截断音频")
        return waveform[:, :target_samples]
    
    # 如果音频时长 < 目标时长，需要循环填充
    print(f"[XVideoMergeNode] 音频长度 < 目标长度，循环填充")
    return loop_audio_safe(waveform, target_samples, sample_rate)


def process_all_videos(video_files, audio_file, tracker):
    """
    处理所有视频文件，添加音频
    
    Args:
        video_files: 视频文件列表
        audio_file: 音频文件路径（可以为 None）
        tracker: 进度追踪器
    
    Returns:
        list: 处理后的视频文件路径列表
    """
    processed_videos = []
    total_files = len(video_files)
    
    # 如果只有一个视频且不需要处理音频，直接使用原文件
    if total_files == 1 and audio_file is None:
        print("[XVideoMergeNode] 只有一个视频且无需音频处理，直接使用")
        return video_files
    
    for idx, video_path in enumerate(video_files):
        tracker.update(description=f"处理视频 {idx+1}/{total_files}")
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_output = f.name
        
        # 如果有音频文件，为每个视频添加音频
        if audio_file:
            # 提取当前视频时长
            duration = get_video_duration(video_path)
            print(f"[XVideoMergeNode] 视频 {idx+1} 时长: {duration:.2f}秒")
            
            # 从总音频中提取对应片段
            with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as f:
                segment_audio = f.name
            
            try:
                # 计算音频片段位置
                start_time = sum(get_video_duration(v) for v in video_files[:idx])
                print(f"[XVideoMergeNode] 音频片段起始: {start_time:.2f}秒, 时长: {duration:.2f}秒")
                
                # 使用 ffmpeg 提取音频片段
                ffmpeg_cmd = get_ffmpeg_path()
                subprocess.run([
                    ffmpeg_cmd,
                    '-i', audio_file,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    '-y',
                    segment_audio
                ], check=True, capture_output=True)
                
                # 处理视频
                process_video_file(video_path, segment_audio, temp_output)
                
            finally:
                # 清理片段音频
                if os.path.exists(segment_audio):
                    os.remove(segment_audio)
        else:
            # 只复制视频，移除音频
            process_video_file(video_path, None, temp_output)
        
        processed_videos.append(temp_output)
        print(f"[XVideoMergeNode] 视频 {idx+1}/{total_files} 已处理: {os.path.basename(temp_output)}")
    
    return processed_videos


def cleanup_temp_files(file_paths, audio_file=None):
    """
    清理临时文件
    
    Args:
        file_paths: 要清理的文件路径列表
        audio_file: 音频文件路径（可选）
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[XVideoMergeNode] 清理临时文件: {file_path}")
            except Exception as e:
                print(f"[XVideoMergeNode] 清理临时文件失败: {e}")
    
    if audio_file and os.path.exists(audio_file):
        try:
            os.remove(audio_file)
            print(f"[XVideoMergeNode] 清理音频文件: {audio_file}")
        except Exception as e:
            print(f"[XVideoMergeNode] 清理音频文件失败: {e}")


class VideoMergeNode:
    """ComfyUI 节点：合并视频并替换音频（带进度显示）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "例如: ./videos 或 C:/videos"
                }),
                "output_file": ("STRING", {
                    "default": "merged_output.mp4",
                    "multiline": False,
                    "placeholder": "输出文件名（相对路径自动拼接到 video_dir）"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "default": None,
                }),
                "wait": ("*", {"default": None, "forceInput": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_file",)
    FUNCTION = "merge"
    CATEGORY = "xy/video"
    
    def merge(self, video_dir, output_file, audio=None, wait=None):
        """
        执行视频合并
        
        Args:
            video_dir: 视频目录
            output_file: 输出文件名
            audio: 可选的音频输入 (AUDIO 类型)
            wait: 可选的等待输入（用于控制流程）
        """
        # ===== 输入验证 =====
        if not video_dir or not video_dir.strip():
            raise ValueError("video_dir 不能为空")
        
        if not output_file or not output_file.strip():
            raise ValueError("output_file 不能为空")
        
        # ===== 处理路径 =====
        app_output_dir = folder_paths.get_output_directory()
        print(f"[XVideoMergeNode] ComfyUI 输出目录: {app_output_dir}")
        
        # 去掉 video_dir 最后一个 / 后面的内容，然后拼接到输出目录
        video_dir_clean = video_dir.strip().rstrip('/\\')
        video_dir_parent = os.path.dirname(video_dir_clean)
        full_video_dir = os.path.join(app_output_dir, video_dir_parent)
        print(f"[XVideoMergeNode] 完整视频目录: {full_video_dir}")
        
        # 处理 output_file
        output_file_clean = output_file.strip()
        
        # ===== 执行合并 =====
        try:
            result = merge_videos_with_audio(
                video_dir=full_video_dir,
                audio_clip=audio,
                output_file=output_file_clean,
                progress_callback=self
            )
            return (result,)
        except Exception as e:
            raise RuntimeError(f"视频合并失败: {str(e)}")


# ===== ComfyUI 节点注册 =====
NODE_CLASS_MAPPINGS = {
    "XVideoMergeNode": VideoMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XVideoMergeNode": "X Video Merge",
}