
from .nodes import NODE_CLASS_MAPPINGS as nodes_mappings, NODE_DISPLAY_NAME_MAPPINGS as nodes_display
from .deepseek import NODE_CLASS_MAPPINGS as deepseek_mappings, NODE_DISPLAY_NAME_MAPPINGS as deepseek_display



# 合并两个模块的映射字典
NODE_CLASS_MAPPINGS = {**nodes_mappings, **deepseek_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**nodes_display, **deepseek_display}

