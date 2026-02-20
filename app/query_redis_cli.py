from fastapi import HTTPException


async def fetch_file_by_user_id(redis_cli, user_id: int) -> str:
    if not isinstance(user_id, int) or user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    active_id_key = f"active_id:{user_id}"
    file_id: str = await redis_cli.get(active_id_key)
    if not file_id:
        raise HTTPException(status_code=404, detail="No active file found for user")
    return file_id

async def fetch_image_by_file_key(redis_cli, file_id: str) -> bytes:
    if not file_id:
        raise HTTPException(status_code=400, detail="File ID is required")

    if not isinstance(file_id, str):
        raise HTTPException(status_code=400, detail="File ID must be a string")

    image_key = f"image_data:{file_id}"
    image_bytes: bytes = await redis_cli.get(image_key)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Image not found")

    return image_bytes
