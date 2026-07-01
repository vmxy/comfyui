import torch
import numpy as np


class IsNotEmptyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": { # 放入 optional
                "value": ("*", {"default": None, "forceInput": False}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_not_empty",)
    FUNCTION = "is_not_empty"
    CATEGORY = "xy/tool"
    
    def is_not_empty(self, value=None):
        """检查值是否非空（反向检查）"""
        check_node = IsEmptyNode()
        is_empty = check_node.is_empty(value)[0]
        return (not is_empty,)


class IsEmptyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "value": ("*", {"default": None, "forceInput": False}),
                #"strict_mode": ("BOOLEAN", {"default": True}),  # 可选参数
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_empty",)
    FUNCTION = "is_empty"
    CATEGORY = "xy/tool"
    
    def is_empty(self, value=None, strict_mode=True):
        """检查值是否为空"""
        is_empty = False
        
        # None 检查
        if value is None:
            is_empty = True
        
        # 字符串检查
        elif isinstance(value, str):
            if strict_mode:
                is_empty = len(value.strip()) == 0
            else:
                is_empty = len(value) == 0
        
        # 列表、元组、字典检查
        elif isinstance(value, (list, tuple, dict, set)):
            is_empty = len(value) == 0
        
        # PyTorch 张量检查
        elif isinstance(value, torch.Tensor):
            is_empty = value.numel() == 0  # 元素总数为0
        
        # NumPy 数组检查
        elif isinstance(value, np.ndarray):
            is_empty = value.size == 0
        
        # 布尔值本身
        elif isinstance(value, bool):
            is_empty = not value  # False 视为空
        
        # 数字：0 是否视为空？可以根据需求调整
        elif isinstance(value, (int, float)):
            is_empty = value == 0  # 可选，如果不想要可以注释掉
            
        # 其他类型
        else:
            # 尝试获取长度
            try:
                is_empty = len(value) == 0
            except TypeError:
                # 无法获取长度，默认 False
                is_empty = False
        
        return (is_empty,)


class SwitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "optional": {  # 关键：if_true 和 if_false 放在 optional 中
                "in_true": ("*", {"default": None, "forceInput": True}),
                "in_false": ("*", {"default": None, "forceInput": True}),
            },
             "required": {
                "condition": ("BOOLEAN", {"default": False, "forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = "xy/tool"
    
    def switch(self, condition, in_true=None, in_false=None):
        """如果 condition 为 True，返回 in_true，否则返回 in_false"""
        return (in_true if condition else in_false,)

class NullNode:
    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "required": {},
        }
    
    RETURN_TYPES = ("None",)
    RETURN_NAMES = ("result",)
    FUNCTION = "handle"
    CATEGORY = "xy/tool"
    
    def handle(self):
        return None


# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "XIsEmpty": IsEmptyNode,
    "XIsNotEmpty": IsNotEmptyNode,
    "XSwitch": SwitchNode,
    "XNull": NullNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "XIsEmpty": "Check IsEmpty",
    "XIsNotEmpty": "Check if Not Empty",
    "XSwitch": "XSwitch",
    "XNull": "XNull XNone",
}
