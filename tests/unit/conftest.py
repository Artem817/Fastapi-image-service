import os
import secrets
import io
import warnings
from typing import List, Tuple

import pytest
from PIL import Image
from pydantic import BaseModel, field_validator
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("DATABASE_URL", os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:"))
os.environ.setdefault("SECRET_KEY", os.getenv("TEST_SECRET_KEY", secrets.token_urlsafe(32)))

warnings.filterwarnings(
    "ignore",
    message=".*crypt.*deprecated.*",
    category=DeprecationWarning,
    module="passlib.*",
)
warnings.filterwarnings(
    "ignore",
    message="Please use `import python_multipart` instead.",
    category=PendingDeprecationWarning,
    module="starlette.formparsers",
)

from app.main import app
from app.database.database import get_redis_text, get_redis_binary
from app.auth.auth import get_current_user

@pytest.fixture
async def ac():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
        
@pytest.fixture
def mock_redis_t():
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.incr.return_value = 1
    mock.expire.return_value = True
    mock.ttl.return_value = 0
    
    app.dependency_overrides[get_redis_text] = lambda: mock
    yield mock
    del app.dependency_overrides[get_redis_text]

@pytest.fixture
def mock_redis_b():
    mock = AsyncMock()
    app.dependency_overrides[get_redis_binary] = lambda: mock
    yield mock
    del app.dependency_overrides[get_redis_binary]

@pytest.fixture
def auth_user():
    user = MagicMock()
    user.id = 817
    user.email = "artem@test.com"
    
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    del app.dependency_overrides[get_current_user]

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
