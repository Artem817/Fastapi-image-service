from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError
from jose import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database.database import get_db, settings
from app.models.models import User
from app.utility.log.log_root import log_ctx
from app.utility.schemas.schemas import TokenData

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, username: str, password: str):
    log = log_ctx(component="auth", username=username)
    user = get_user_by_username(db, username)
    if not user:
        log.warning("auth_failed_user_not_found")
        return False
    if not verify_password(password, user.hashed_password):
        log.warning("auth_failed_bad_password")
        return False
    log.info("auth_success", extra={"user_id": user.id})
    return user


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
):
    log = log_ctx(component="auth")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        username = payload.get("sub")
        if username is None:
            log.warning("token_missing_subject")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        log.warning("token_invalid")
        raise credentials_exception

    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        log.warning("token_user_not_found", extra={"username": token_data.username})
        raise credentials_exception
    return user
