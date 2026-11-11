"""SQLAlchemy database engine and session management for SPECTRA."""
from __future__ import annotations

from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.core.config import settings

# Ensure storage directory exists before creating SQLite file
_ = settings.storage_path

connect_args = {"check_same_thread": False, "timeout": 30} if settings.SPECTRA_DB_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.SPECTRA_DB_URL,
    connect_args=connect_args,
    future=True,
)

if settings.SPECTRA_DB_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    import app.database.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
