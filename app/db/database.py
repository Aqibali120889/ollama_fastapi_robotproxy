from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings


def build_engine(settings: Settings) -> AsyncEngine:
    # A short connect timeout matters here specifically because every DB call
    # in the live voice pipeline goes through app.db.safe_logging's
    # never-raise wrappers -- if Postgres is unreachable, those must fail
    # fast, not hang for the OS-level TCP timeout (tens of seconds), which
    # would eat directly into the voice pipeline's 4s latency budget.
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"timeout": settings.database_connect_timeout_seconds},
    )


def build_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    engine = build_engine(settings)
    return async_sessionmaker(engine, expire_on_commit=False)
