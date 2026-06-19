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
        
    try:
        is_alive = await redis_text.ping()
        if not is_alive:
            raise AppError("Redis ping failed", status_code=503)
            
        return {"redis_status": "online"}
        
    except Exception as e:
        log.error(f"redis_probe_failed: {str(e)}")
        raise AppError("Failed to connect to Redis", status_code=503)