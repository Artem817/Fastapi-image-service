from PIL import Image
from enum import Enum
from fastapi import Query, HTTPException, Depends

class AspectRatio(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_4_3 = "4:3"
    RATIO_1_1 = "1:1"
    RATIO_21_9 = "21:9"
    RATIO_9_16 = "9:16"
    RATIO_4_5 = "4:5"
    RATIO_2_3 = "2:3"

ASPECT_RATIO_MAP = {
    AspectRatio.RATIO_16_9: (16, 9),
    AspectRatio.RATIO_4_3: (4, 3),
    AspectRatio.RATIO_1_1: (1, 1),
    AspectRatio.RATIO_21_9: (21, 9),
    AspectRatio.RATIO_9_16: (9, 16),
    AspectRatio.RATIO_4_5: (4, 5),
    AspectRatio.RATIO_2_3: (2, 3),
}

Image.MAX_IMAGE_PIXELS = 64_000_000  # 8000 x 8000
MAX_SIZE = 20 * 1024 * 1024
IMAGE_EXTENSION_ALIASES = {
    "jpeg": "jpg",
    "jpe": "jpg",
    "tiff": "tif",
}
PIL_FORMAT_TO_EXTENSION = {
    "JPEG": "jpg",
    "JPG": "jpg",
    "PNG": "png",
    "GIF": "gif",
    "WEBP": "webp",
    "TIFF": "tif",
    "BMP": "bmp",
}
EXTENSION_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}

# RESIZE_ASPECT_RATIO = {
#     "width": lambda target_w, orig_h, target_ratio: (
#         target_w, 
#         int(target_w / target_ratio) 
#     ),
#     "height": lambda target_h, orig_w, target_ratio: (
#         int(target_h * target_ratio), 
#         target_h
#     )
# }

