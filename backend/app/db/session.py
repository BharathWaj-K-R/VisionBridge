"""
SQLAlchemy engine + session factory.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

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


@event.listens_for(Session, "after_begin")
def _set_visionbridge_rls_context(session, transaction, connection):
    """Bind the authenticated FastAPI user to the current DB transaction.

    The value is transaction-local so pooled connections cannot retain one
    request's identity for the next request. SQLite has no custom PostgreSQL
    setting support, so local tests simply skip this hook.
    """
    if connection.dialect.name != "postgresql":
        return

    user_id = session.info.get("visionbridge_user_id")
    if user_id is None:
        return

    connection.execute(
        text("SELECT set_config('app.user_id', :user_id, true)"),
        {"user_id": str(user_id)},
    )


def get_db():
    """FastAPI dependency: yields a DB session and guarantees it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
