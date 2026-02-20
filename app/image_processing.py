from PIL import Image, ImageOps
from abc import ABC, abstractmethod
import io

from .models_unet import model_arch
from .watermark_tool import WatermarkEngine
import tempfile
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np


def output_transform_bytes(image: Image.Image, output_format: str) -> bytes:
    if output_format.upper() in ["JPEG", "JPG"] and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    with io.BytesIO() as output:
        image.save(output, format=output_format)
        return output.getvalue()


class ImageProcessingStrategy(ABC):
    @abstractmethod
    def process_image(self, image_bytes: bytes) -> bytes:
        pass

class RotateImageStrategy(ImageProcessingStrategy):
    def __init__(self, angle: int = 45, output_format: str = "PNG"):
        self.angle = angle % 360
        self.output_format = output_format

    def process_image(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rotated = image.rotate(self.angle, expand=True)
            return output_transform_bytes(rotated, self.output_format)


class FlipImageStrategy(ImageProcessingStrategy):
    def __init__(self, direction: str = "horizontal", output_format: str = "PNG"):
        self.direction = direction if direction in ["horizontal", "vertical"] else "horizontal"
        self.output_format = output_format

    def process_image(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if self.direction == "horizontal":
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

            return output_transform_bytes(flipped_image, self.output_format)

class ChangeFormatStrategy(ImageProcessingStrategy):
    def __init__(self, target_format: str = "JPEG"):
        self.target_format = target_format.upper() if target_format.upper() in ["JPEG", "PNG", "WEBP"] else "JPEG"

    def process_image(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return output_transform_bytes(image, self.target_format)

class ImageProcessor:
    def __init__(self, strategy: ImageProcessingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ImageProcessingStrategy):
        self.strategy = strategy

    def process_image(self, image_bytes: bytes) -> bytes:
        return self.strategy.process_image(image_bytes)

class BasicFilterStrategy(ImageProcessingStrategy):
    def __init__(self, filter_type: str = "normal"):
        self.filter_type = filter_type if filter_type in ["grayscale", "sepia", "posterize", "invert"] else "normal"
        self._filter_handlers = {
            "posterize": lambda img: img.posterize(4),
            "grayscale": lambda img: img.convert("L"),
            "sepia": lambda img: img.convert("RGB",
                                             (0.393, 0.769, 0.189, 0, 0.349, 0.686, 0.168, 0, 0.272, 0.534, 0.131, 0)),
            "invert": lambda img: ImageOps.invert(img),
            "normal": lambda img: img
        }

    def process_image(self, image_bytes: bytes) -> bytes:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                filtered_image = self._filter_handlers[self.filter_type](image)
                return output_transform_bytes(filtered_image, image.format)
        except Exception as e:
            return image_bytes

class ResizeStrategy(ImageProcessingStrategy):
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height

    def process_image(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            resized_image = image.resize((self.width, self.height))
            return output_transform_bytes(resized_image, image.format)

class RemoveBackground(ImageProcessingStrategy):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as image:
            original_img = image.convert("RGB")
            w, h = original_img.size
            
            input_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = model_arch.model(input_tensor)
                pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            mask_img = Image.fromarray((pred_mask * 255).astype('uint8'), 'L')
            mask_img = mask_img.resize((w, h), Image.BILINEAR)
            
            result_img = original_img.copy()
            result_img.putalpha(mask_img)
            
            return output_transform_bytes(result_img, "PNG")
        
class WatermarkStrategy(ImageProcessingStrategy):
    def __init__(
        self,
        logo_path: str,
        opacity: float = 0.4,
        rotation: int = 30,
        scale_percent: float = 0.2,
        density: float = 2.0,
        randomize: bool = True,
        jitter: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.logo_path = logo_path
        self.opacity = opacity
        self.rotation = rotation
        self.scale_percent = scale_percent
        self.density = density
        self.randomize = randomize
        self.jitter = jitter
        self.seed = seed

    def process_image(self, image_bytes: bytes) -> bytes:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                tmp_img.write(image_bytes)
                tmp_img_path = tmp_img.name
            
            engine = WatermarkEngine(
                opacity=self.opacity,
                rotation=self.rotation,
                scale_percent=self.scale_percent,
                density=self.density,
                randomize=self.randomize,
                jitter=self.jitter,
                seed=self.seed,
            )
            
            result_image = engine.apply(tmp_img_path, self.logo_path)
            
            with io.BytesIO() as output:
                result_image.save(output, format="PNG")
                processed_bytes = output.getvalue()
            
            Path(tmp_img_path).unlink()
            
            return processed_bytes
        except Exception as e:
            print(f"Watermark error: {e}")
            return image_bytes
