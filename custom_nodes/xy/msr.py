import cv2
import numpy as np
import torch
from PIL import Image


class MSRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 736, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 32, "max": 8192, "step": 32}),
                "frame_count": ([17, 25, 33, 41], {"default": 17}),
            },
            "optional": {
                "1": ("IMAGE",),
                "2": ("IMAGE",),
                "3": ("IMAGE",),
                "4": ("IMAGE",),
                "background": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("output", "frame_count")
    FUNCTION = "create_video"
    CATEGORY = "xy/msr"

    def create_video(self, width, height, frame_count, background=None, **kwargs):
        if background is None:
            raise ValueError("background input is required")

        images = []
        for name in ("1", "2", "3", "4"):
            image = kwargs.get(name)
            if image is not None:
                images.append(self._prepare_image(image, (width, height)))

        images.append(self._prepare_image(background, (width, height)))
        frames = self._expand_frames(images, frame_count)
        output = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        return (output, frame_count)

    @staticmethod
    def _tensor_to_rgb_array(image):
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        return np.ascontiguousarray(image)

    @staticmethod
    def _prepare_image(image, target_size):
        image_array = MSRNode._tensor_to_rgb_array(image)
        pil_image = Image.fromarray(image_array).convert("RGB")
        image_array = np.array(pil_image)
        if image_array.shape[1] == target_size[0] and image_array.shape[0] == target_size[1]:
            return np.ascontiguousarray(image_array)
        #return cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)
         # 使用新的resize方法，保持宽高比并裁剪
        return MSRNode._resize_with_crop(image_array, target_size)

    @staticmethod
    def _resize_with_crop(image_array, target_size):
        """
        保持宽高比resize，然后中心裁剪到目标尺寸
        
        Args:
            image_array: numpy数组格式的图像 (H, W, C)
            target_size: 目标尺寸 (width, height)
        
        Returns:
            裁剪后的图像 (height, width, C)
        """
        target_w, target_h = target_size
        h, w = image_array.shape[:2]
        
        # 计算缩放比例，保持宽高比
        target_ratio = target_w / target_h
        src_ratio = w / h
        
        if src_ratio > target_ratio:
            # 原图更宽 → 按高度缩放，宽度会超出
            new_h = target_h
            new_w = int(target_h * src_ratio)
        else:
            # 原图更高或相等 → 按宽度缩放，高度会超出
            new_w = target_w
            new_h = int(target_w / src_ratio)
        
        # resize到中间尺寸（保持宽高比）
        resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 中心裁剪到目标尺寸
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
        
        return np.ascontiguousarray(cropped)
        
    @staticmethod
    def _expand_frames(images, frame_count):
        base_count = frame_count // len(images)
        remainder = frame_count % len(images)
        frames = []
        for index, image in enumerate(images):
            repeats = base_count + (1 if index < remainder else 0)
            frames.extend([image] * repeats)
        return frames

NODE_CLASS_MAPPINGS = {
    "XMSRNode": MSRNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XMSRNode": "X Image Concat 组合 MSR",
}
