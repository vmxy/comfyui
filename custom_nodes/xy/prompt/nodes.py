#from comfy_api.latest import io

class PromptParse:
    """Encodes temporal local prompts and patches the model for Prompt Relay."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #"clip": ("CLIP",),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "local_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }
    
    #RETURN_TYPES = ("CONDITIONING", "STRING", "LIST")
    #RETURN_NAMES = ("positive", "global_prompt", "local_list")
    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("global_prompt", "local_list")
    OUTPUT_IS_LIST = (False, False)  # 第三个输出是列表
    FUNCTION = "execute"
    CATEGORY = "xy/prompt"

    def execute(self, global_prompt, local_prompt):
        locals_list = [p.strip() for p in local_prompt.split("|") if p.strip()]
        if not locals_list:
            raise ValueError("At least one local prompt is required (separate with |)")
                
        #full_prompt = f"{global_prompt}\r\n{locals_list[local_index]}"
        #conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        
        return (global_prompt, locals_list)



class PromptEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"default": None, "forceInput": True}),
                "global_prompt": ("STRING", {"default": None, "forceInput": True}),
                "locals_list": ("LIST", {"default": None, "forceInput": True}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "prompt")
    FUNCTION = "encode"
    CATEGORY = "xy/prompt"

    def encode(self, clip, global_prompt, locals_list, index):
        prompt = self.get(locals_list, index)
        print(f"local prompt={prompt}")
        full_prompt = f"{global_prompt}\r\n{prompt}"
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        return (conditioning, full_prompt) 
    
    def get(self, array, index):
        # 处理各种可索引类型
        if array is None:
            return (None,)
        print(f"test type = {type(array)}")  # 应该是 <class 'list'>
        print(f"locals_list={len(array)}")
        print(f"view: {array}")
        # 支持元组、列表、字符串
        try:
            if isinstance(array, (tuple, list)):
                # 如果索引超出范围，返回 None 或最后一个元素
                if index >= len(array):
                    return array[-1] if len(array) > 0 else ""
                return array[index]
            elif isinstance(array, str):
                # 字符串按字符取
                if index >= len(array):
                    return array[-1] if len(array) > 0 else ""
                return array[index]
            else:
                # 不支持的类型，直接返回原值
                return array
        except (TypeError, IndexError):
            return array
    
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
    


NODE_CLASS_MAPPINGS = {
    "XPromptParse": PromptParse,
    "XPromptEncode": PromptEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XPromptParse": "XPrompt Parse",
    "XPromptEncode": "XPrompt Encode",
}
