from fastapi import APIRouter, Depends
from ..database.database import get_redis_text
from ..utility.exceptions.exceptions import AppError
from ..utility.log.log_root import log_ctx
from ..utility.schemas.schemas import ErrorResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/ping-redis",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
)
async def ping_redis(redis_text=Depends(get_redis_text)):
    log = log_ctx(endpoint="ping-redis")
    
    log.info("request_received")
    if not redis_text:
        log.error("redis_not_connected")
        raise AppError("Redis not connected", status_code=503)
    val = await redis_text.get("some_key")
    if val is None:
        log.error("redis_probe_failed")
        raise AppError("Failed to retrieve value from Redis", status_code=503)
    return {"redis_status": "online", "value": val}
