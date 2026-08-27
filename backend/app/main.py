from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, calibration, dashboard, evaluation, health, history, translate, users
from app.core.config import get_settings
from app.db.session import Base, engine

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistence only while the application is running."""
    settings.validate_for_runtime()
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add conservative response headers at the API boundary."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(translate.router, prefix=settings.API_V1_PREFIX)
app.include_router(calibration.router, prefix=settings.API_V1_PREFIX)
app.include_router(dashboard.router, prefix=settings.API_V1_PREFIX)
app.include_router(history.router, prefix=settings.API_V1_PREFIX)
app.include_router(users.router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {"message": "VisionBridge API — see /docs for the interactive API explorer"}
