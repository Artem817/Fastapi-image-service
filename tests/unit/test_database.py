import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.database import Base, make_async_database_url, make_sync_database_url
from app.models.models import User

pytestmark = pytest.mark.anyio


def test_database_url_helpers_normalize_drivers():
    assert (
        make_async_database_url("postgresql://user:pass@localhost:5432/app")
        == "postgresql+asyncpg://user:pass@localhost:5432/app"
    )
    assert (
        make_sync_database_url("postgresql+asyncpg://user:pass@localhost:5432/app")
        == "postgresql+psycopg2://user:pass@localhost:5432/app"
    )


async def test_async_session_can_write_and_read_user():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(User(username="db-user", hashed_password="hash"))
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(select(User).where(User.username == "db-user"))
            user = result.scalar_one()

        assert user.username == "db-user"
        assert user.is_active is True
    finally:
        await engine.dispose()
