import pytest
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import io
from pydantic import BaseModel, field_validator


@pytest.fixture
def shelf_image(request):
    params = getattr(request, "param", {})
    config = ShelfImageConfig(**params)
    factory = ShelfImageFactory(config)
    buf = factory.create_standart()
    yield buf
    buf.close()
    

class ShelfImageConfig(BaseModel):
    rgb: str = "RGB"
    color: Tuple[int, int, int] = (255, 0, 0)
    size: Tuple[int, int] = (100, 100)
    fmt: str = "PNG"
    
    @field_validator('color', 'size', mode='before')
    @classmethod
    def validate_tuples(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v

class ShelfImageFactory:
    def __init__(self, config: ShelfImageConfig):
        self.config = config
    
    def create_config_img(self):
        return Image.new(self.config.rgb, size=self.config.size, color=self.config.color)
        
    def create_standart(self) -> io.BytesIO:
        buf = io.BytesIO()
        image = Image.new(self.config.rgb, self.config.size, self.config.color)
        image.save(buf, format=self.config.fmt)
        buf.seek(0)
        return buf
    
    def create_pixels_colors(
        self, 
        colors: List[Tuple[int, int, int]],
        direction: str = "horizontal"
    ) -> io.BytesIO:
        """Create a test image with exactly two pixels of specified colors at specified coordinates.

        This method is used for deterministic testing of spatial image transformations
        (flip, rotate, etc.), verifying that specific pixels end up in expected positions
        after processing.

        Args:
            colors: List of exactly 2 tuples with (r, g, b) color values.

        Returns:
            io.BytesIO: Image buffer with the specified pixels.

        Raises:
            ValueError: If coordinates or colors don't contain exactly 2 tuples.

        Note:
            Avoids byte comparison or golden image management by using pixel
            coordinate and color verification instead.
        """
        if len(colors) != 2:
            raise ValueError("colors must contain exactly 2 tuples: [(r,g,b), (r,g,b)]")
        
        img = self.create_config_img()
        buf = io.BytesIO()
        img.putpixel((0, 0), colors[0])
        img.putpixel((1, 0), colors[1])
        img.save(buf, format=self.config.fmt)
        buf.seek(0)
        return buf
    
        
        