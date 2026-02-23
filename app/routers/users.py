from fastapi import APIRouter, Depends

from ..auth import get_current_user
from ..log_root import log_ctx
from ..models import User

router = APIRouter(tags=["users"])


@router.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    log = log_ctx(endpoint="profile", user_id=current_user.id)
    log.info("request_received")
    return {
        "id": current_user.id,
        "username": current_user.username,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
    }
