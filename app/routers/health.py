from fastapi import APIRouter, Depends

from ..database import get_redis_text

router = APIRouter(tags=["health"])


@router.get("/ping-redis")
async def ping_redis(redis_text=Depends(get_redis_text)):
    if not redis_text:
        return {"error": "Redis not connected"}
    val = await redis_text.get("some_key")
    return {"redis_status": "online", "value": val}
