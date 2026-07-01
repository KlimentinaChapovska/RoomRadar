"""Health-check endpoints."""
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models import model_store
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health():
    """Always returns 200 — confirms the process is alive."""
    info = model_store.get_load_info()
    return {
        "status": "ok",
        "models_loaded": info["loaded"],
        "load_time_s": info["load_time_s"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready", summary="Readiness probe")
async def ready():
    """Returns 200 when models are loaded; 503 otherwise."""
    ts = datetime.now(timezone.utc).isoformat()
    if model_store.is_ready():
        return {"status": "ready", "models_loaded": True, "timestamp": ts}

    info = model_store.get_load_info()
    detail = (
        info["load_error"]
        or "Models not loaded — run ml/scripts/finalize_classification.py "
           "and ml/scripts/finalize_regression.py, then restart the server."
    )
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "models_loaded": False,
            "detail": detail,
            "timestamp": ts,
        },
    )
