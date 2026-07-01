"""RoomRadar FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.middleware.logging import LoggingMiddleware
from app.models import model_store
from app.routers import health, predict
from app.routers import model_info
from app.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("RoomRadar API starting up — loading models...")
    model_store.load_models()
    if model_store.is_ready():
        info = model_store.get_load_info()
        log.info("Startup complete — models ready in %.3fs.", info["load_time_s"])
    else:
        info = model_store.get_load_info()
        log.warning("Startup complete — models NOT ready: %s", info["load_error"])
    yield
    log.info("RoomRadar API shutting down.")


_cors_raw     = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()] or ["*"]

app = FastAPI(
    title="RoomRadar API",
    description="Hotel cancellation and room price prediction API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")
app.include_router(model_info.router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )
