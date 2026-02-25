import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

from abc import ABC, abstractmethod
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import numpy as np

from app.models_unet import model_arch
from app.services.watermark_tool import WatermarkEngine
from app.utility.constants import ASPECT_RATIO_MAP, AspectRatio
from app.utility.log.log_root import log_ctx

logger = logging.getLogger(__name__)

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

def processing_log(operation: str, **fields) -> logging.LoggerAdapter:
    return log_ctx(component="image_processing", operation=operation, **fields)

class RotateImageStrategy(ImageProcessingStrategy):
    def __init__(self, angle: int = 45, output_format: str = "PNG"):
        self.angle = angle % 360
        self.output_format = output_format

    def process_image(self, image_bytes: bytes) -> bytes:
        log = processing_log("rotate", angle=self.angle, output_format=self.output_format)
        log.info("processing_start")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                rotated = image.rotate(self.angle, expand=True)
                processed = output_transform_bytes(rotated, self.output_format)
            log.info("processing_success")
            return processed
        except Exception:
            log.exception("processing_failed")
            raise


class FlipImageStrategy(ImageProcessingStrategy):
    def __init__(self, direction: str = "horizontal", output_format: str = "PNG"):
        self.direction = direction if direction in ["horizontal", "vertical"] else "horizontal"
        self.output_format = output_format

    def process_image(self, image_bytes: bytes) -> bytes:
        log = processing_log("flip", direction=self.direction, output_format=self.output_format)
        log.info("processing_start")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if self.direction == "horizontal":
                    flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                else:
                    flipped_image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                processed = output_transform_bytes(flipped_image, self.output_format)
            log.info("processing_success")
            return processed
        except Exception:
            log.exception("processing_failed")
            raise

class ChangeFormatStrategy(ImageProcessingStrategy):
    def __init__(self, target_format: str = "JPEG"):
        self.target_format = target_format.upper() if target_format.upper() in ["JPEG", "PNG", "WEBP"] else "JPEG"

    def process_image(self, image_bytes: bytes) -> bytes:
        log = processing_log("change_format", target_format=self.target_format)
        log.info("processing_start")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                processed = output_transform_bytes(image, self.target_format)
            log.info("processing_success")
            return processed
        except Exception:
            log.exception("processing_failed")
            raise

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
        log = processing_log("filter", filter_type=self.filter_type)
        log.info("processing_start")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                filtered_image = self._filter_handlers[self.filter_type](image)
                processed = output_transform_bytes(filtered_image, image.format or "PNG")
            log.info("processing_success")
            return processed
        except Exception as e:
            log.exception("processing_failed")
            raise ValueError(f"Failed to apply filter {self.filter_type}") from e

class ResizeStrategy(ImageProcessingStrategy):
    def __init__(self, ratio_enum: AspectRatio, req_width: int):
        self.ratio_enum = ratio_enum
        self.req_width = req_width

    def process_image(self, image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        w_factor, h_factor = ASPECT_RATIO_MAP[self.ratio_enum]
        
        target_width = min(self.req_width, orig_w)
        target_width = max(1, target_width)

        target_height = max(1, int((target_width * h_factor) / w_factor))

        log = processing_log("resize", width=target_width, height=target_height)
        log.info("processing_start")

        try:
            resized_img = ImageOps.fit(img, (target_width, target_height), centering=(0.5, 0.5))
            processed = output_transform_bytes(resized_img, resized_img.format or "PNG")
            
            log.info("processing_success")
            return processed
        except Exception:
            log.exception("processing_failed")
            raise
class RemoveBackground(ImageProcessingStrategy):
    def __init__(self, model: Optional[torch.nn.Module] = None, device: Optional[str] = None, threshold: float = 0.5):
        if model is None:
            self.model = model_arch.get_loaded_model()
        else:
            self.model = model

        if hasattr(self.model, "eval"):
            self.model.eval()

        if device:
            self.device = torch.device(device)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        else:
            try:
                self.device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")

        self.threshold = threshold
        self.transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_bytes):
        log = processing_log(
            "remove_bg",
            device=str(self.device),
            threshold=self.threshold,
        )
        log.info("processing_start")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                original_img = image.convert("RGB")
                w, h = original_img.size

                input_tensor = self.transform(original_img).unsqueeze(0).to(self.device)  # type: ignore

                with torch.no_grad():
                    logits = self.model(input_tensor)
                    pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()

                mask_normalized = (pred_mask * 255).astype("uint8")
                mask_img = Image.fromarray(mask_normalized).convert("L")
                resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR  # type: ignore
                mask_img = mask_img.resize((w, h), resample)

                cutoff = int(self.threshold * 255)
                mask_img = mask_img.point(lambda p: 255 if p >= cutoff else 0) # type: ignore

                result_img = original_img.convert("RGBA")
                result_img.putalpha(mask_img)

                processed = output_transform_bytes(result_img, "PNG")
            log.info("processing_success", extra={"width": w, "height": h})
            return processed
        except Exception:
            log.exception("processing_failed")
            raise
        
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
        log = processing_log(
            "watermark",
            opacity=self.opacity,
            rotation=self.rotation,
            scale_percent=self.scale_percent,
            density=self.density,
            randomize=self.randomize,
            jitter=self.jitter,
            seed=self.seed,
        )
        log.info("processing_start")
        tmp_img_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                tmp_img.write(image_bytes)
                tmp_img_path = tmp_img.name

            engine = WatermarkEngine(
                opacity=self.opacity,
                rotation=self.rotation,
                scale_percent=self.scale_percent,
                density=int(self.density),
                randomize=self.randomize,
                jitter=self.jitter,
                seed=self.seed,
            )

            result_image = engine.apply(tmp_img_path, self.logo_path)

            with io.BytesIO() as output:
                result_image.save(output, format="PNG")
                processed_bytes = output.getvalue()

            log.info("processing_success")
            return processed_bytes
        except Exception:
            log.exception("processing_failed")
            raise
        finally:
            if tmp_img_path:
                Path(tmp_img_path).unlink(missing_ok=True)
