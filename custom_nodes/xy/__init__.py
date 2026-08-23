# 导入 xy 和 image 两个子模块的映射
from .xy import NODE_CLASS_MAPPINGS as xy_mappings, NODE_DISPLAY_NAME_MAPPINGS as xy_display
from .msr import NODE_CLASS_MAPPINGS as msr_mappings, NODE_DISPLAY_NAME_MAPPINGS as msr_display
from .video import NODE_CLASS_MAPPINGS as video_mappings, NODE_DISPLAY_NAME_MAPPINGS as video_display
from .prompt import NODE_CLASS_MAPPINGS as prompt_mappings, NODE_DISPLAY_NAME_MAPPINGS as prompt_display
from .asr import NODE_CLASS_MAPPINGS as asr_mappings, NODE_DISPLAY_NAME_MAPPINGS as asr_display

# 合并两个模块的映射字典
NODE_CLASS_MAPPINGS = {**xy_mappings, **msr_mappings, **video_mappings, **prompt_mappings, **asr_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**xy_display, **msr_display, **video_display, **prompt_display, **asr_display}

# 声明模块导出的公共接口
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]