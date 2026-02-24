from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.auth import (
    authenticate_user,
    create_access_token,
    get_password_hash,
    settings,
)
from app.database.database import get_db
from app.models.models import User
from app.utility.schemas.schemas import Token, UserCreate, UserLogin, UserResponse
from app.utility.log.log_root import log_ctx
router = APIRouter(tags=["auth"])


@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    log = log_ctx(endpoint="register", username=user.username)
    log.info("request_received")

    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        log.warning("username_already_registered")
        raise HTTPException(
            status_code=400,
            detail="Username already registered",
        )

    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    log.info("register_success", extra={"user_id": db_user.id})

    return db_user


@router.post("/login", response_model=Token)
def login_user(user_credentials: UserLogin, db: Session = Depends(get_db)):
    log = log_ctx(endpoint="login", username=user_credentials.username)
    log.info("request_received")
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        log.warning("login_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires,
    )
    log.info("login_success", extra={"user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}
