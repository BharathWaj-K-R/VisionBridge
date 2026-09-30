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

if settings.ENV.lower() == "production" and settings.DATABASE_URL.startswith("postgresql"):
    @event.listens_for(engine, "connect")
    def _set_production_database_role(dbapi_connection, connection_record):
        """Drop the production connection into the least-privileged app role."""
        cursor = dbapi_connection.cursor()
        cursor.execute("SET ROLE visionbridge_app")
        cursor.close()


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

    settings_to_set = {}
    user_id = session.info.get("visionbridge_user_id")
    if user_id is not None:
        settings_to_set["app.user_id"] = str(user_id)

    operation = session.info.get("visionbridge_auth_operation")
    if operation:
        settings_to_set["app.auth_operation"] = operation

    identifier = session.info.get("visionbridge_auth_identifier")
    if identifier is not None:
        settings_to_set["app.auth_identifier"] = identifier

    username = session.info.get("visionbridge_auth_username")
    if username is not None:
        settings_to_set["app.auth_username"] = username

    email = session.info.get("visionbridge_auth_email")
    if email is not None:
        settings_to_set["app.auth_email"] = email

    for key, value in settings_to_set.items():
        connection.execute(
            text("SELECT set_config(:key, :value, true)"),
            {"key": key, "value": value},
        )


def get_db():
    """FastAPI dependency: yields a DB session and guarantees it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
