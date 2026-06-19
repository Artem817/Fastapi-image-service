from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import make_url

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings() # type: ignore

class Base(DeclarativeBase):
    pass

def make_async_database_url(url: str) -> str:
    parsed = make_url(url)
    drivername = parsed.drivername
    if drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+asyncpg")
    elif drivername == "sqlite":
        parsed = parsed.set(drivername="sqlite+aiosqlite")
    return parsed.render_as_string(hide_password=False)


def make_sync_database_url(url: str) -> str:
    parsed = make_url(url)
    drivername = parsed.drivername
    if drivername == "postgresql+asyncpg":
        parsed = parsed.set(drivername="postgresql+psycopg2")
    elif drivername == "sqlite+aiosqlite":
        parsed = parsed.set(drivername="sqlite")
    return parsed.render_as_string(hide_password=False)


async_database_url = make_async_database_url(settings.database_url)
engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
if async_database_url.startswith("sqlite+aiosqlite:///:memory:"):
    engine_kwargs = {
        "connect_args": {"check_same_thread": False},
        "poolclass": StaticPool,
    }

engine = create_async_engine(async_database_url, **engine_kwargs)
SessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

async def get_redis_text(request: Request):
    return request.app.state.redis

async def get_redis_binary(request: Request):
    return request.app.state.redis_binary
