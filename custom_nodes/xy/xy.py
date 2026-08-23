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
            "optional": {
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
        return (in_true if condition else in_false,)
    
    @classmethod
    def get_output_types(cls, input_types, input_values):
        """根据输入值推断输出类型"""
        if "in_true" in input_values and input_values["in_true"] is not None:
            return (type(input_values["in_true"]).__name__,)
        elif "in_false" in input_values and input_values["in_false"] is not None:
            return (type(input_values["in_false"]).__name__,)
        return ("*",)
        


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



class GetItem:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "array": ("*", {"default": None, "forceInput": True}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("result",)
    FUNCTION = "get"
    CATEGORY = "xy/tool"
    
    def get(self, array, index):
        # 处理各种可索引类型
        if array is None:
            return (None,)
        #print(f"test type = {type(array)}")  # 应该是 <class 'list'>
        #print(f"locals_list={len(array)}")
        #print(f"view: {array}")
        # 支持元组、列表、字符串
        try:
            if isinstance(array, (tuple, list)):
                # 如果索引超出范围，返回 None 或最后一个元素
                if index >= len(array):
                    return (array[-1] if len(array) > 0 else None,)
                return (array[index],)
            elif isinstance(array, str):
                # 字符串按字符取
                if index >= len(array):
                    return (array[-1] if len(array) > 0 else "",)
                return (array[index],)
            else:
                # 不支持的类型，直接返回原值
                return (array,)
        except (TypeError, IndexError):
            return (array,)
    
    @classmethod
    def get_output_types(cls, input_types, input_values):
        """推断输出类型"""
        # 获取 array 的实际类型
        if "array" in input_types:
            array_type = input_types["array"]
            if array_type and array_type != "*":
                # 如果是列表或元组，尝试提取元素类型
                if array_type.startswith("list") or array_type.startswith("tuple"):
                    # 简单情况下返回元素类型，复杂情况保持 *
                    return ("*",)
                return (array_type,)
        return ("*",)
    

class BasenameNode:
    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "required": {
                "path": ("STRING", {"default": None, "forceInput": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("name",)
    FUNCTION = "handle"
    CATEGORY = "xy/tool"
    
    def handle(self, path):
        return (os.path.basename(path))


# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "XIsEmpty": IsEmptyNode,
    "XIsNotEmpty": IsNotEmptyNode,
    "XSwitch": SwitchNode,
    "XNull": NullNode,
    "XGetItem": GetItem,
    "XBasename": BasenameNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "XIsEmpty": "Check IsEmpty",
    "XIsNotEmpty": "Check if Not Empty",
    "XSwitch": "XSwitch",
    "XNull": "XNull XNone",
    "XGetItem": "XGet Item",
    "XBasename": "XBasename",
}
