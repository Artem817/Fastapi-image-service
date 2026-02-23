import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis import asyncio as aioredis

from .database import engine
from .models import Base
from .routers import auth as auth_router
from .routers.exceptions_handler import register_exception_handlers
from .routers import health as health_router
from .routers import images as images_router
from .routers import users as users_router
from .log_root import setup_logging
from .log_root import log_ctx

Base.metadata.create_all(bind=engine)
setup_logging()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log = log_ctx(component="lifespan")
    log.info("startup_begin")
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis_binary = await aioredis.from_url(REDIS_URL, decode_responses=False)
    log.info("startup_complete")

    yield

    log.info("shutdown_begin")
    await app.state.redis.close()
    await app.state.redis_binary.close()
    log.info("shutdown_complete")


async def get_user_settings(redis, user_id: int):
    key = f"user_config:{user_id}"
    data = await redis.get(key)

    if data:
        return json.loads(data)
    else:
        default_settings = {"filter": "normal", "intensity": 1.0, "crop": None}
        await save_user_settings(redis, user_id, default_settings)
        return default_settings


async def save_user_settings(redis, user_id: int, settings: dict):
    key = f"user_config:{user_id}"
    await redis.set(key, json.dumps(settings), ex=3600)


app = FastAPI(title="Image Management API", version="1.0.0", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(health_router.router)
app.include_router(auth_router.router)
app.include_router(images_router.router)
app.include_router(users_router.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
