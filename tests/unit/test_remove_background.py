import io

import pytest
import torch
from PIL import Image

from app.image_processing import ImageProcessor, RemoveBackground


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
        # FakeModel emits logits; large positive => mask ~1, large negative => mask ~0
        ({"size": (4, 4), "color": (0, 0, 255)}, (4, 4), 10.0, 255, (0, 0, 255)),
        ({"size": (3, 2), "color": (255, 0, 0)}, (3, 2), -10.0, 0, (255, 0, 0)),
    ],
    indirect=["shelf_image"],
    ids=["keep_all", "remove_all"],
)
def test_remove_background_respects_model_mask(shelf_image, expected_size, mask_value, expected_alpha, expected_color):
    shelf_bytes = shelf_image.getvalue()

    # Use a fake mask to avoid asserting behavior of a human-segmentation model on synthetic images
    processor = ImageProcessor(RemoveBackground(model=FakeModel(mask_value)))
    processed_bytes = processor.process_image(shelf_bytes)

    with Image.open(io.BytesIO(processed_bytes)) as img:
        img = img.convert("RGBA")

        assert img.size == expected_size, f"The size should remain {expected_size}, but it is {img.size}"

        alpha_channel = img.getchannel("A")
        alpha_min, alpha_max = alpha_channel.getextrema()

        assert alpha_min == expected_alpha and alpha_max == expected_alpha, (
            f"Alpha channel should reflect mask value {mask_value}; got range {alpha_min}-{alpha_max}"
        )

        # When mask keeps everything, pixel colors must be preserved
        if expected_alpha == 255:
            assert img.getpixel((0, 0)) == (*expected_color, 255)
