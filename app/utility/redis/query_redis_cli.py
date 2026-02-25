from fastapi import HTTPException
from ..log.log_root import log_ctx


async def fetch_file_by_user_id(redis_cli, user_id: int) -> str:
    log = log_ctx(component="redis_query", user_id=user_id)
    if not isinstance(user_id, int) or user_id <= 0:
        log.warning("invalid_user_id")
        raise HTTPException(status_code=400, detail="Invalid user ID")

    active_id_key = f"active_id:{user_id}"
    file_id: str = await redis_cli.get(active_id_key)
    if not file_id:
        log.info("active_file_missing")
        raise HTTPException(status_code=404, detail="No active file found for user")
    log.info("active_file_found", extra={"file_id": file_id})
    return file_id

async def fetch_image_by_file_key(redis_cli, file_id: str) -> bytes:
    log = log_ctx(component="redis_query", file_id=file_id)
    if not file_id:
        log.warning("file_id_missing")
        raise HTTPException(status_code=400, detail="File ID is required")

    if not isinstance(file_id, str):
        log.warning("file_id_invalid_type")
        raise HTTPException(status_code=400, detail="File ID must be a string")

    image_key = f"image_data:{file_id}"
    image_bytes: bytes = await redis_cli.get(image_key)
    if not image_bytes:
        log.info("image_missing")
        raise HTTPException(status_code=404, detail="Image not found")

    log.info("image_found")
    return image_bytes
