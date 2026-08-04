

class SplitSpeaker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            		"audio": ("AUDIO", {
                    "default": None,
                    "placeholder": "说话人音频"

                }),
                "text": ("STRING", {
                		"default": None,
                		"placeholder": "说话文本:格式: [speaker_x]: xxx 格式"
                }),
                "index": ("INT", {
						"default": 0,
                		"placeholder": "说话文本:格式: [speaker_x]: xxx 格式"
                })
            },
            "optional": { # 放入 optional
                #"value": ("*", {"default": None, "forceInput": False}),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("is_not_empty",)
    FUNCTION = "is_not_empty"
    CATEGORY = "xy/tts"
    
    def split(self, audio, text):
        """检查值是否非空（反向检查）"""
        
        return ("",)

# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "XSplitSpeaker": SplitSpeaker,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "XSplitSpeaker": "X Split tts speaker",
}
