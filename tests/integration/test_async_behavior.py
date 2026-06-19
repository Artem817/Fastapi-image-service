import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.database.database import get_redis_text
from app.main import app

pytestmark = pytest.mark.anyio


class SlowAsyncRedis:
    async def get(self, key: str) -> str:
        await asyncio.sleep(0.1)
        return f"value-for-{key}"

    async def ping(self) -> bool:
        await asyncio.sleep(0.1)  
        return True


@pytest.fixture
async def client_with_slow_redis():
    redis = SlowAsyncRedis()
    app.dependency_overrides[get_redis_text] = lambda: redis
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.pop(get_redis_text, None)


async def test_ping_redis_handles_concurrent_requests(client_with_slow_redis):
    start = time.perf_counter()

    responses = await asyncio.gather(
        *[client_with_slow_redis.get("/ping-redis") for _ in range(5)]
    )

    elapsed = time.perf_counter() - start
    assert [response.status_code for response in responses] == [200, 200, 200, 200, 200]
    
    assert elapsed < 0.35