from openai import OpenAI
import os



class DeepSeek:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "deepseek-v4-flash",
                        "deepseek-v4-pro",
                    ],
                    {"default": "deepseek-v4-flash"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "xy/prompt"

    def generate(
        self,
        api_key,
        system_prompt,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DEEPSEEK_API_KEY") is not None:
            API_KEY = os.getenv("DEEPSEEK_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}],
            stream=False
            )
            
        return (completion.choices[0].message.content,)



class DeepSeekAceStep:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "deepseek-v4-flash",
                        "deepseek-v4-pro",
                    ],
                    {"default": "deepseek-v4-flash"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "song")
    FUNCTION = "generate"
    CATEGORY = "xy/prompt"

    def generate(
        self,
        api_key,
        system_prompt,
        prompt,
        model,
        seed,
    ):

        if os.getenv("DEEPSEEK_API_KEY") is not None:
            API_KEY = os.getenv("DEEPSEEK_API_KEY")
        elif api_key.strip() != "":
            API_KEY = api_key
        else:
            raise ValueError("API Key is not set")
                
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}],
            stream=False
            )
        ks = completion.choices[0].message.content.split("|") 
        tags = ks[0] if len(ks) > 0 else ""
        song = ks[1] if len(ks) > 1 else ""
        return (tags, song)



NODE_CLASS_MAPPINGS = {
    "XDeepSeek": DeepSeek,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XDeepSeek": "XDeepSeek DeepSeek",
}
