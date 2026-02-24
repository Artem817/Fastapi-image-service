import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis import asyncio as aioredis

from app.models_unet import model_arch

from app.database.database import engine
from app.models.models import Base
from app.routers import auth as auth_router
from app.routers.exceptions_handler import register_exception_handlers
from app.routers import health as health_router
from app.routers import images as images_router
from app.routers import users as users_router
from app.utility.log.log_root import setup_logging
from app.utility.log.log_root import log_ctx
from fastapi import Request

Base.metadata.create_all(bind=engine)
setup_logging()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def get_model(request: Request):
    return request.app.state.segmentation_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    log = log_ctx(component="lifespan")
    log.info("startup_begin")
    
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis_binary = await aioredis.from_url(REDIS_URL, decode_responses=False)
    
    log.info("model_loading_start")
    try:
        app.state.segmentation_model = model_arch.get_loaded_model()
        if hasattr(app.state.segmentation_model, "eval"):
            app.state.segmentation_model.eval()
        log.info("model_loading_complete")
    except Exception as e:
        log.exception("model_loading_failed", extra={"error": str(e)})
        raise RuntimeError("Failed to load ML model") from e

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
