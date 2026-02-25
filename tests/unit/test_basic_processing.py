from typing import List, Tuple
import pytest
import io
from PIL import Image

from app.services.image_processing import FlipImageStrategy, ImageProcessor, ResizeStrategy, ChangeFormatStrategy, RotateImageStrategy, output_transform_bytes
from app.utility.constants import AspectRatio
from tests.unit.conftest import ShelfImageFactory, ShelfImageConfig
BLACK = (0, 0, 0)
RED = (255, 0, 0)

@pytest.mark.parametrize(
    "shelf_image, fmt, signature",
    [
        ({"color": (0, 0, 255)}, "PNG", b"\x89PNG"),
        ({"color": (255, 0, 0)}, "JPEG", b"\xff\xd8"),
        ({"color": (0, 0, 0)}, "JPEG", b"\xff\xd8")
    ],
    indirect=["shelf_image"],
    ids=["PNG_Blue", "JPEG_Red", "JPEG_Black"] 
)
def test_output_transform_bytes_formats(shelf_image, fmt, signature):
    with Image.open(shelf_image) as image:
        result = output_transform_bytes(image, fmt)
        assert isinstance(result, bytes)
        assert result.startswith(signature)
        
        
@pytest.mark.parametrize(
    "shelf_image, angle, expected_size",
    [
        ({"size": (100, 50)}, 90, (50, 100)),
        ({"size": (100, 50)}, 180, (100, 50)),
    ],
    indirect=["shelf_image"],
    ids=["rotate_90", "rotate_180"]
)
def test_rotate_dimensions(shelf_image, angle, expected_size):
    shelf_bytes = shelf_image.getvalue()
    
    processor = ImageProcessor(RotateImageStrategy(angle=angle))
    processed_bytes = processor.process_image(shelf_bytes)
    
    with Image.open(io.BytesIO(processed_bytes)) as img:
        assert img.size == expected_size

@pytest.mark.parametrize(
    "shelf_image, ratio, req_width, expected_w, expected_h",
    [
        # 1. Standard Square: 100x100 -> 1:1, req 50 => 50x50
        ({"size": (100, 100)}, AspectRatio.RATIO_1_1, 50, 50, 50),
        
        # 2. Aspect Ratio Change: 100x100 -> 16:9, req 80 => 80x45 (80 * 9 / 16)
        ({"size": (100, 100)}, AspectRatio.RATIO_16_9, 80, 80, 45),
        
        # 3. Portrait Ratio: 100x100 -> 2:3, req 60 => 60x90 (60 * 3 / 2)
        ({"size": (100, 100)}, AspectRatio.RATIO_2_3, 60, 60, 90),
        
        # 4. Upscale Protection: 100x100 -> 1:1, req 500 => 100x100 (the original is taken)
        ({"size": (100, 100)}, AspectRatio.RATIO_1_1, 500, 100, 100),
        
        # 5. Small Image Wide Ratio: 100x50 -> 21:9, req 200 => 100x42 (min(200, 100)=100, 100*9/21=42)
        ({"size": (100, 50)}, AspectRatio.RATIO_21_9, 200, 100, 42),
        
        # 6. Minimum sizes: 1x1 -> 4:3, req 10 => 1x0 (Pillow may throw an error, або 1x1)
        ({"size": (1, 1)}, AspectRatio.RATIO_4_3, 10, 1, 0), 
    ],
    indirect=["shelf_image"],
    ids=["square_downscale", "to_wide_screen", "to_portrait", "upscale_protection", "wide_limit", "micro_image"]
)
def test_resize_strategy_logic(shelf_image, ratio, req_width, expected_w, expected_h):
    """
    Testing the resize strategy:
    - Height calculation by coefficient – is it correct?
    - Does the upscale protection work (min(req_width, orig_w))”
    """
    shelf_bytes = shelf_image.getvalue()
    
    strategy = ResizeStrategy(ratio_enum=ratio, req_width=req_width)
    processor = ImageProcessor(strategy)
    
    processed_bytes = processor.process_image(shelf_bytes)
    
    with Image.open(io.BytesIO(processed_bytes)) as img:
        assert img.size[0] == expected_w
        assert abs(img.size[1] - expected_h) <= 1
        
@pytest.mark.parametrize(
    "shelf_image, target_format, expected_signature",
    [
        ({"size": (100, 100), "fmt": "PNG"}, "JPEG", b"\xff\xd8"),
        ({"size": (150, 100), "fmt": "JPEG"}, "PNG", b"\x89PNG"),
        ({"size": (100, 150), "fmt": "WEBP"}, "JPEG", b"\xff\xd8"),
        ({"size": (2, 1), "fmt": "PNG"}, "JPEG", b"\xff\xd8"),
        ({"size": (3, 3), "fmt": "JPEG"}, "PNG", b"\x89PNG"),
        ({"size": (1, 1), "fmt": "WEBP"}, "PNG", b"\x89PNG"),
    ],
    indirect=["shelf_image"],
    ids=["PNG_to_JPEG", "JPEG_to_PNG", "WEBP_to_JPEG", "small_PNG_to_JPEG", "small_JPEG_to_PNG", "small_WEBP_to_PNG"]
)
def test_change_format(shelf_image, target_format, expected_signature):
    shelf_bytes = shelf_image.getvalue()
    processor = ImageProcessor(ChangeFormatStrategy(target_format=target_format))
    processed_bytes = processor.process_image(shelf_bytes)
    
    assert isinstance(processed_bytes, bytes)
    assert processed_bytes.startswith(expected_signature), \
        f"Expected format {target_format} signature {expected_signature}, got {processed_bytes[:4]}"

@pytest.mark.parametrize(
    "direction, expected_size, pixel_checks",
    [
        (
            "horizontal",
            (2, 1),
            [
                ((0, 0), RED),
                ((1, 0), BLACK),
            ]
        ),
        (
            "vertical",
            (2, 1),
            [
                ((0, 0), BLACK),
                ((1, 0), RED),
            ]
        ),
    ],
    ids=["flip_horizontal", "flip_vertical"]
)
def test_flip_logic(direction, expected_size, pixel_checks):
    config = ShelfImageConfig(size=(2, 1), color=(255, 255, 255))
    factory = ShelfImageFactory(config)
    shelf_image = factory.create_pixels_colors([BLACK, RED])
    shelf_bytes = shelf_image.getvalue()
    
    strategy = FlipImageStrategy(direction)
    processor = ImageProcessor(strategy)
    processed_bytes = processor.process_image(shelf_bytes)
    
    with Image.open(io.BytesIO(processed_bytes)) as img:
        assert img.size == expected_size
        for pixel_coord, expected_color in pixel_checks:
            actual_color = img.getpixel(pixel_coord)
            assert actual_color == expected_color, \
                f"Pixel at {pixel_coord} should be {expected_color}, got {actual_color}"
