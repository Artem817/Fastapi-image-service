import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database.database import Base, get_db
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
async def test_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield session_factory
    finally:
        await engine.dispose()


@pytest.fixture
async def client_with_db(test_session_factory):
    async def override_get_db():
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.pop(get_db, None)


async def test_register_login_and_profile(client_with_db):
    register_response = await client_with_db.post(
        "/register",
        json={"username": "alice", "password": "correct-horse"},
    )

    assert register_response.status_code == 200
    registered = register_response.json()
    assert registered["username"] == "alice"

    login_response = await client_with_db.post(
        "/login",
        json={"username": "alice", "password": "correct-horse"},
    )

    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    profile_response = await client_with_db.get(
        "/profile",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert profile_response.status_code == 200
    assert profile_response.json()["username"] == "alice"
