import os
import secrets
import warnings

import pytest

os.environ.setdefault("DATABASE_URL", os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:"))
os.environ.setdefault("SECRET_KEY", os.getenv("TEST_SECRET_KEY", secrets.token_urlsafe(32)))
os.environ.setdefault("MODEL_URL", "")

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


@pytest.fixture
def anyio_backend():
    return "asyncio"
