"""Model metadata endpoint — returns safe public fields only."""
from fastapi import APIRouter, HTTPException

from app.models import model_store
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["models"])

# Fields exposed to API consumers.  Internal training artefacts (hyperparameters,
# dataset row counts, validation-set comparisons, feature importances) are excluded
# to avoid exposing implementation details that could help game the models.
_CLF_PUBLIC_FIELDS = {
    "model_name", "version", "trained_on", "positive_class",
    "threshold", "calibration_method", "risk_bands",
    "test_metrics", "features_required", "limitations",
}

_REG_PUBLIC_FIELDS = {
    "model_name", "version", "trained_on", "target",
    "target_units", "target_transformation", "predict_inverse_transform",
    "test_metrics", "dummy_baseline", "features_required", "limitations",
}


def _filter(meta: dict, allowed: set) -> dict:
    return {k: v for k, v in meta.items() if k in allowed}


@router.get("/model-info", summary="Return public metadata for all loaded models")
async def model_info():
    """Returns safe public metadata for the cancellation and price models."""
    if not model_store.is_ready():
        info = model_store.get_load_info()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "models_not_ready",
                "message": info["load_error"] or "Models are not loaded.",
            },
        )

    return {
        "cancellation_model": _filter(
            model_store.get_cancellation_metadata(), _CLF_PUBLIC_FIELDS
        ),
        "price_model": _filter(
            model_store.get_price_metadata(), _REG_PUBLIC_FIELDS
        ),
    }
