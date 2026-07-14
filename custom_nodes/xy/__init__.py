# 导入 xy 和 image 两个子模块的映射
from .xy import NODE_CLASS_MAPPINGS as xy_mappings, NODE_DISPLAY_NAME_MAPPINGS as xy_display
from .image import NODE_CLASS_MAPPINGS as image_mappings, NODE_DISPLAY_NAME_MAPPINGS as image_display

# 合并两个模块的映射字典
NODE_CLASS_MAPPINGS = {**xy_mappings, **image_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**xy_display, **image_display}

# 声明模块导出的公共接口
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]