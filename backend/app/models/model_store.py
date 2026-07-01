"""
Singleton model store — loads both production models once at startup.

Both PKL files use custom sklearn transformers that must be registered in
the import namespace BEFORE joblib.load() is called, otherwise pickle
fails with AttributeError.  The sys.path insertion and noqa imports do
exactly that.
"""
import json
import sys
import time
from pathlib import Path
from typing import Optional

import joblib

ROOT = Path(__file__).resolve().parents[3]  # backend/app/models/ → FinalProject/
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

# Register custom transformers before any joblib.load call
from calibration_pipeline import CalibratedPipeline  # noqa: F401
from train_classification import FeatureEngineer      # noqa: F401
from train_regression import RegFeatureEngineer       # noqa: F401

from app.utils.logger import get_logger

log = get_logger("model_store")

_MODELS_DIR = ROOT / "models"

_store: dict = {
    "cancellation_model":    None,
    "cancellation_metadata": None,
    "price_model":           None,
    "price_metadata":        None,
    "loaded":                False,
    "load_error":            None,
    "load_time_s":           None,
}


def load_models() -> None:
    """Load both production models from disk.  Idempotent — safe to call multiple times."""
    if _store["loaded"]:
        log.debug("Models already loaded — skipping reload.")
        return

    t0 = time.perf_counter()
    errors: list[str] = []

    for name, pkl_name, meta_name in [
        ("cancellation", "cancellation_model.pkl",  "cancellation_metadata.json"),
        ("price",        "room_price_model.pkl",     "room_price_metadata.json"),
    ]:
        pkl_path  = _MODELS_DIR / pkl_name
        meta_path = _MODELS_DIR / meta_name
        try:
            _store[f"{name}_model"]    = joblib.load(pkl_path)
            _store[f"{name}_metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
            log.info("Loaded %s model  (%s)", name, pkl_path.name)
        except FileNotFoundError as exc:
            msg = f"Model file not found: {exc.filename}"
            errors.append(msg)
            log.error(msg)
        except Exception as exc:
            msg = f"Failed to load {name} model: {exc}"
            errors.append(msg)
            log.error(msg, exc_info=True)

    elapsed = round(time.perf_counter() - t0, 3)
    _store["load_time_s"] = elapsed

    if errors:
        _store["loaded"]     = False
        _store["load_error"] = "; ".join(errors)
        log.error("Model loading FAILED (%d error(s)) in %.3fs", len(errors), elapsed)
    else:
        _store["loaded"]     = True
        _store["load_error"] = None
        log.info("All models loaded successfully in %.3fs", elapsed)


def is_ready() -> bool:
    return bool(_store["loaded"])


def get_cancellation_model():
    return _store["cancellation_model"]


def get_price_model():
    return _store["price_model"]


def get_cancellation_metadata() -> Optional[dict]:
    return _store["cancellation_metadata"]


def get_price_metadata() -> Optional[dict]:
    return _store["price_metadata"]


def get_load_info() -> dict:
    return {
        "loaded":      _store["loaded"],
        "load_error":  _store["load_error"],
        "load_time_s": _store["load_time_s"],
    }
