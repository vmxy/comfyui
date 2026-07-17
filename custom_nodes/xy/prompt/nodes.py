#from comfy_api.latest import io

class PromptEncode:
    """Encodes temporal local prompts and patches the model for Prompt Relay."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "local_prompts": ("STRING", {"multiline": True, "default": ""}),
                "local_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING", "LIST")
    RETURN_NAMES = ("positive", "global_prompt", "local_list")
    OUTPUT_IS_LIST = (False, False, False)  # 第三个输出是列表
    FUNCTION = "execute"
    CATEGORY = "xy/prompt"

    def execute(self, clip, global_prompt, local_prompts, local_index):
        locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
        if not locals_list:
            raise ValueError("At least one local prompt is required (separate with |)")
        
        print(f"locals_list={len(locals_list)}")
        print(f"view: {locals_list}")
        
        full_prompt = f"{global_prompt}\r\n{locals_list[local_index]}"
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        
        return (conditioning, global_prompt, locals_list)

NODE_CLASS_MAPPINGS = {
    "XPromptEncode": PromptRelayEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XPromptEncode": "XPrompt Encode",
}
