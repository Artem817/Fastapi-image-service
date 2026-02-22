import io
import os
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import filetype
from ..auth import get_current_user
from ..database import get_redis_binary, get_redis_text
from ..image_processing import (
    BasicFilterStrategy,
    FlipImageStrategy,
    ImageProcessor,
    ResizeStrategy,
    RotateImageStrategy,
    WatermarkStrategy,
    RemoveBackground,
)
from ..exceptions import (
    AppError,
    ImageNotFoundError,
    InvalidFileError,
    ModelNotAvailableError,
    RateLimitExceededError,
)
from ..models import ImageStore, User
from ..query_redis_cli import fetch_file_by_user_id
from ..schemas import WatermarkPhotoRequest

Image.MAX_IMAGE_PIXELS = 89478485
MAX_SIZE = 10 * 1024 * 1024

router = APIRouter(prefix="/images", tags=["images"])

async def rate_limit_upload(redis_text, user_id: int, max_uploads: int = 5) -> None:
    """
    Checks the user's hourly upload limit.
        
    Args:
        redis_text: Redis client
        user_id: User ID
        max_uploads: Maximum uploads per hour
        
    Raises:
        RateLimitExceededError: If the limit is exceeded
    """
    key = f"rate_upload:{user_id}"
    count = await redis_text.incr(key)
    
    if count == 1:
        await redis_text.expire(key, 3600)
    
    if count > max_uploads:
        ttl = await redis_text.ttl(key)
        raise RateLimitExceededError(max_uploads=max_uploads, retry_after=ttl)

def load_image_info(image_bytes: bytes) -> tuple[str | None, tuple[int, int]]:
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        return image.format, image.size

def save_locally(current_user: User, file_id: str, image_bytes: bytes) -> None:
    folder_path = "processed_images"
    os.makedirs(folder_path, exist_ok=True)
    filename = f"processed_image_{current_user.id}_{file_id}.png"
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "wb") as f:
        f.write(image_bytes)

def guess_media_type(data: bytes):
    if data.startswith(b'\xff\xd8\xff'):
        return "image/jpeg", "jpg"
    elif data.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png", "png"
    elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
        return "image/gif", "gif"
    return "application/octet-stream", "bin"

async def fetch_active_image(redis_text, redis_binary, user_id: int) -> tuple[str, bytes]:
    try:
        file_id = await fetch_file_by_user_id(redis_text, user_id)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            detail = exc.detail if isinstance(exc.detail, str) else None
            raise ImageNotFoundError(user_id=user_id, detail=detail) from exc
        raise
    image_bytes = await redis_binary.get(f"image_data:{file_id}")
    if not image_bytes:
        raise ImageNotFoundError(file_id=file_id, detail="Image data lost")
    return file_id, image_bytes


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    active_id_key = f"active_id:{current_user.id}"
    file_id = await redis_text.get(active_id_key)
    if file_id:
        return {"status": "already_uploaded", "file_id": file_id}

    await rate_limit_upload(redis_text, current_user.id, max_uploads=5)

    image_bytes = await file.read(MAX_SIZE + 1)
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    kind = filetype.guess(image_bytes)
    image_type = kind.extension if kind else None
    if image_type is None:
        raise InvalidFileError("Invalid image file")

    try:
        image_format, image_size = await run_in_threadpool(load_image_info, image_bytes)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidFileError("Invalid image file") from exc

    img_store = ImageStore()
    file_id = str(img_store.id)

    set_result = await redis_text.set(
        f"active_id:{current_user.id}", file_id, ex=3600, nx=True
    )
    if not set_result:
        existing = await redis_text.get(active_id_key)
        return {"status": "already_uploaded", "file_id": existing}

    await redis_binary.set(f"image_data:{file_id}", image_bytes, ex=3600)

    return {
        "status": "uploaded",
        "file_id": file_id,
        "format": image_format,
        "size": image_size,
    }
    
@router.post("/filter")
async def filter_image(
    filter_type: str = "normal",
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    file_id, image_bytes = await fetch_active_image(redis_text, redis_binary, current_user.id)
    processor = ImageProcessor(BasicFilterStrategy(filter_type=filter_type))
    processed_bytes = await run_in_threadpool(processor.process_image, image_bytes)
    await run_in_threadpool(save_locally, current_user, file_id, processed_bytes)
    await redis_binary.set(f"image_data:{file_id}", processed_bytes, ex=3600)
    return {"status": "success", "message": "Image filtered"}


@router.post("/resize")
async def resize_endpoint(
    width: int = 256,
    height: int = 256,
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    file_id, image_bytes = await fetch_active_image(redis_text, redis_binary, current_user.id)
    processor = ImageProcessor(ResizeStrategy(width=width, height=height))

    processed_bytes = await run_in_threadpool(processor.process_image, image_bytes)

    await run_in_threadpool(save_locally, current_user, file_id, processed_bytes)
    await redis_binary.set(f"image_data:{file_id}", processed_bytes, ex=3600)

    return {"status": "success", "message": "Image resized"}


# FIXME: /images/clarity is not implemented yet. Uncomment when ready.
# @router.post("/clarity")
# async def image_clarity(
#     output_format: str = "PNG",
#     current_user: User = Depends(get_current_user),
#     redis_text=Depends(get_redis_text),
#     redis_binary=Depends(get_redis_binary),
# ):
#     raise HTTPException(
#         status_code=status.HTTP_501_NOT_IMPLEMENTED,
#         detail="Image clarity feature is not yet implemented",
#     )

@router.post("/flip")
async def flip_image(
    direction: str = "horizontal",
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    file_id, image_bytes = await fetch_active_image(redis_text, redis_binary, current_user.id)
    processor = ImageProcessor(FlipImageStrategy(direction=direction))
    processed_bytes = await run_in_threadpool(processor.process_image, image_bytes)
    await run_in_threadpool(save_locally, current_user, file_id, processed_bytes)
    await redis_binary.set(f"image_data:{file_id}", processed_bytes, ex=3600)
    return {"status": "success", "message": "Image flipped"}


@router.post("/rotate")
async def rotate_image(
    angle: int,
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    file_id, image_bytes = await fetch_active_image(redis_text, redis_binary, current_user.id)
    processor = ImageProcessor(RotateImageStrategy(angle=angle))
    processed_bytes = await run_in_threadpool(processor.process_image, image_bytes)

    await run_in_threadpool(save_locally, current_user, file_id, processed_bytes)

    await redis_binary.set(f"image_data:{file_id}", processed_bytes, ex=3600)

    return {"status": "success", "message": f"Rotated by {angle}"}


@router.post("/watermark")
async def watermark_image(
    watermark_params: WatermarkPhotoRequest = Depends(WatermarkPhotoRequest.as_form),
    logo_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    if not logo_file.content_type or not logo_file.content_type.startswith("image/"):
        raise AppError("Watermark logo must be an image", status_code=400)

    file_id, image_bytes = await fetch_active_image(
        redis_text, redis_binary, current_user.id
    )

    logo_bytes = await logo_file.read()
    if not logo_bytes:
        raise InvalidFileError("Watermark logo file is empty")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_logo:
        tmp_logo.write(logo_bytes)
        logo_path = tmp_logo.name

    try:
        processor = ImageProcessor(
            WatermarkStrategy(
                logo_path=logo_path,
                opacity=watermark_params.opacity,
                rotation=watermark_params.rotation,
                scale_percent=watermark_params.scale_percent,
                density=watermark_params.density,
                randomize=watermark_params.randomize,
                jitter=watermark_params.jitter,
                seed=watermark_params.seed,
            )
        )
        processed_bytes = await run_in_threadpool(processor.process_image, image_bytes)
        await run_in_threadpool(save_locally, current_user, file_id, processed_bytes)
        await redis_binary.set(f"image_data:{file_id}", processed_bytes, ex=3600)

        return {
            "status": "success",
            "message": "Watermark applied successfully",
            "opacity": watermark_params.opacity,
            "rotation": watermark_params.rotation,
            "scale": watermark_params.scale_percent,
            "density": watermark_params.density,
            "randomize": watermark_params.randomize,
            "jitter": watermark_params.jitter,
            "seed": watermark_params.seed,
        }
    finally:
        Path(logo_path).unlink(missing_ok=True)

@router.post("/remove_bg")
async def remove_bg(
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
    ): 
    """
    Remove background from image using ResNet101-UNet model.
    
    This endpoint processes images to remove the background and return an image with 
    transparent background (PNG with alpha channel). The model is trained specifically 
    for human segmentation and works best with photos containing people.
    
    Note: The model weights file (resnet101_unet.pth) must be downloaded separately 
    and placed in the app/models_unet/ directory. The endpoint will return an error 
    if the .pth file is not available.
    
    Args:
        current_user: Current authenticated user
        redis_text: Redis text client for session management
        redis_binary: Redis binary client for image data storage
        
    Returns:
        dict: Status response with success message or error details
        
    Raises:
        HTTPException: If image not found, processing fails, or model is not loaded
        
    Response:
        {
            "status": "success",
            "message": "Background successfully removed"
        }
    """
    file_id, image_bytes = await fetch_active_image(redis_text, redis_binary, current_user.id)
    try:
        processor = ImageProcessor(RemoveBackground())
    except RuntimeError as exc:
        raise ModelNotAvailableError() from exc
    processed_bytes = await run_in_threadpool(processor.process_image, image_bytes)
    await redis_binary.set(f"image_data:{file_id}", processed_bytes, ex=3600)
    return {
            "status": "success",
            "message": "Background successfully removed",
        }


@router.get("/export")
async def export_current_image(
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    file_id, image_bytes = await fetch_active_image(redis_text, redis_binary, current_user.id)
    content_type, extension = guess_media_type(image_bytes)
    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={file_id}.{extension}"}
    )


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    redis_text=Depends(get_redis_text),
    redis_binary=Depends(get_redis_binary),
):
    active_id_key = f"active_id:{current_user.id}"
    file_id = await redis_text.get(active_id_key)
    if not file_id:
        raise ImageNotFoundError(user_id=current_user.id)
    if file_id != image_id:
        raise ImageNotFoundError(file_id=image_id, detail="Image is not active for user")

    await redis_text.delete(active_id_key)
    await redis_binary.delete(f"image_data:{file_id}")
    return {
        "status": "success",
        "message": f"Photo {file_id} has been deleted from memory, {current_user.username}",
    }
    
