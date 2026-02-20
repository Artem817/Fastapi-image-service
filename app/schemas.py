from uuid import UUID

from fastapi import Form
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class ImageID(BaseModel):
    id: int
    uuid: UUID
    created_at: datetime
    updated_at: datetime

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class CropImage(ImageID):
    x: float
    y: float
    width: float
    height: float


class RotatePhotoRequest(ImageID):
    angle: float
    expand: Optional[bool]

class WatermarkPhotoRequest(BaseModel):
    opacity: float = Field(0.4, ge=0.01, le=1.0)
    rotation: int = Field(30, ge=-180, le=180)
    scale_percent: float = Field(0.2, gt=0.0, le=1.0)
    density: float = Field(2.0, gt=0.0, le=10.0)
    randomize: bool = True
    jitter: float = Field(0.2, ge=0.0, le=0.9)
    seed: Optional[int] = None

    @classmethod
    def as_form(
        cls,
        opacity: float = Form(0.4),
        rotation: int = Form(30),
        scale_percent: float = Form(0.2),
        density: float = Form(2.0),
        randomize: bool = Form(True),
        jitter: float = Form(0.2),
        seed: Optional[int] = Form(None),
    ) -> "WatermarkPhotoRequest":
        if opacity > 1.0 and opacity <= 100.0:
            opacity = opacity / 100.0
        return cls(
            opacity=opacity,
            rotation=rotation,
            scale_percent=scale_percent,
            density=density,
            randomize=randomize,
            jitter=jitter,
            seed=seed,
        )

class MirrorPhotoRequest(ImageID):
    mode : str = "horizontal"

class CompressPhotoRequest(ImageID):
    quality: float
    format: str = "JPEG"

class ChangePhotoRequest(ImageID):
    format: str = "PNG"

class FilterPhotoRequest(ImageID):
    filter: str = "grayscale"

class TransformPhotoRequest(ImageID):
    resize: Optional[dict[str, int]]
    crop: Optional[CropImage]
    rotate: Optional[RotatePhotoRequest]
    watermark: Optional[WatermarkPhotoRequest]
    mirror: Optional[MirrorPhotoRequest]
    compress: Optional[CompressPhotoRequest]
    change_format: Optional[ChangePhotoRequest]
    filter: Optional[FilterPhotoRequest]


