import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

from app.image_processing import ImageProcessor, RemoveBackground

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
INPUT_PATH = os.path.join(FIXTURE_DIR, "sample_input.jpg")
EXPECTED_PATH = os.path.join(FIXTURE_DIR, "sample_expected.png")

def calculate_rmse(image_a: Image.Image, image_b: Image.Image):
    """Calculates the root mean square deviation between two images."""
    if image_a.size != image_b.size:
        image_b = image_b.resize(image_a.size)
    
    arr_a = np.array(image_a.convert("RGBA")).astype(np.float32)
    arr_b = np.array(image_b.convert("RGBA")).astype(np.float32)
    
    return np.sqrt(np.mean((arr_a - arr_b) ** 2))

class FakeModel(torch.nn.Module):
    def __init__(self, fill_value: float):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        b, _, h, w = x.shape
        return torch.full((b, 1, h, w), self.fill_value, device=x.device)


@pytest.mark.parametrize(
    "shelf_image, expected_size, mask_value, expected_alpha, expected_color",
    [
        ({"size": (4, 4), "color": (0, 0, 255)}, (4, 4), 10.0, 255, (0, 0, 255)),
        ({"size": (3, 2), "color": (255, 0, 0)}, (3, 2), -10.0, 0, (255, 0, 0)),
    ],
    indirect=["shelf_image"],
    ids=["keep_all", "remove_all"],
)
def test_remove_background_respects_model_mask(shelf_image, expected_size, mask_value, expected_alpha, expected_color):
    shelf_bytes = shelf_image.getvalue()

    processor = ImageProcessor(RemoveBackground(model=FakeModel(mask_value)))
    processed_bytes = processor.process_image(shelf_bytes)

    assert processed_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Expected PNG file signature"

    with Image.open(io.BytesIO(processed_bytes)) as opened:
        assert opened.format == "PNG", f"Expected output format PNG, got {opened.format}"
        assert opened.size == expected_size, f"The size should remain {expected_size}, but it is {opened.size}"
        img = opened.convert("RGBA")

        alpha_channel = img.getchannel("A")
        alpha_min, alpha_max = alpha_channel.getextrema()
        alpha_data = list(alpha_channel.getdata())
        
        if mask_value > 0:
            assert all(v >= 250 for v in alpha_data), "Alpha should be mostly opaque"
        elif mask_value < 0:
            assert all(v <= 5 for v in alpha_data), "Alpha should be mostly transparent"
            
        assert alpha_min == expected_alpha and alpha_max == expected_alpha, (
            f"Alpha channel should reflect mask value {mask_value}; got range {alpha_min}-{alpha_max}"
        )

        if expected_alpha == 255:
            assert img.getpixel((0, 0)) == (*expected_color, 255)
            
@pytest.mark.parametrize("threshold", [0.5])
def test_remove_background_regression(threshold):
    if not os.path.exists(INPUT_PATH):
        pytest.fail(f"Input fixture missing: {INPUT_PATH}")
    
    with open(INPUT_PATH, "rb") as f:
        input_bytes = f.read()

    strategy = RemoveBackground(threshold=threshold)
    processed_bytes = strategy.process_image(input_bytes)

    with Image.open(io.BytesIO(processed_bytes)) as result_img:
        if not os.path.exists(EXPECTED_PATH):
            result_img.save(EXPECTED_PATH)
            pytest.fail(f"Reference image created at {EXPECTED_PATH}. Please verify it and re-run.")

        with Image.open(EXPECTED_PATH) as expected_img:
            rmse_value = calculate_rmse(result_img, expected_img)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert rmse_value < 1.0, f"Image processing drift detected! RMSE: {rmse_value}"
