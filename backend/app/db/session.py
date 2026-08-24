"""
SQLAlchemy engine + session factory.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import get_settings

settings = get_settings()

# check_same_thread=False is needed only for SQLite when used with FastAPI's
# threaded request handling.
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)


if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
        """Make SQLite enforce the foreign keys declared by the ORM models."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and guarantees it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
