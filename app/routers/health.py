from fastapi import APIRouter, Depends
from ..database import get_redis_text
from ..log_root import log_ctx

router = APIRouter(tags=["health"])


@router.get("/ping-redis")
async def ping_redis(redis_text=Depends(get_redis_text)):
    log = log_ctx(endpoint="ping-redis")
    
    log.info("request_received")
    if not redis_text:
        log.error("Redis connection failed")
        return {"error": "Redis not connected"}
    val = await redis_text.get("some_key")
    if val is None:
        log.error("Failed to retrieve value from Redis")
        return {"error": "Failed to retrieve value from Redis"}
    return {"redis_status": "online", "value": val}
