"""
Tests for the saved cancellation production model.
Verifies the pkl loads, metadata is complete, and a real prediction works.
"""
import json, sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

# Register class in the unpickling namespace before loading the model
from calibration_pipeline import CalibratedPipeline  # noqa: F401
from train_classification import FeatureEngineer      # noqa: F401
MODEL_PATH = ROOT / "models" / "cancellation_model.pkl"
META_PATH  = ROOT / "models" / "cancellation_metadata.json"

SAMPLE_BOOKING = {
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 30,
    "arrival_year": 2018,
    "arrival_month": 6,
    "arrival_date": 15,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 99.0,
    "no_of_special_requests": 1,
    # cleaning flags added during preprocessing
    "is_zero_adults": 0,
    "is_zero_nights": 0,
    "price_is_zero": 0,
}

REQUIRED_META_KEYS = {
    "model_name", "version", "trained_on", "positive_class",
    "n_train_rows", "xgb_params", "calibration_method",
    "brier_uncalibrated", "brier_calibrated",
    "threshold", "risk_bands",
    "train_metrics", "val_metrics", "test_metrics",
    "overfitting_notes", "features_required",
}

REQUIRED_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "brier"}


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    assert META_PATH.exists(), f"Metadata not found: {META_PATH}"
    return json.loads(META_PATH.read_text())


@pytest.fixture(scope="module")
def sample_df():
    return pd.DataFrame([SAMPLE_BOOKING])


# ── model file tests ──────────────────────────────────────────────────────────

def test_model_file_exists():
    assert MODEL_PATH.exists(), "cancellation_model.pkl not found in models/"


def test_metadata_file_exists():
    assert META_PATH.exists(), "cancellation_metadata.json not found in models/"


def test_model_file_nonempty():
    assert MODEL_PATH.stat().st_size > 10_000, "Model file is suspiciously small"


# ── metadata tests ────────────────────────────────────────────────────────────

def test_metadata_required_keys(metadata):
    missing = REQUIRED_META_KEYS - set(metadata.keys())
    assert not missing, f"Missing metadata keys: {missing}"


def test_metadata_threshold_valid(metadata):
    t = metadata["threshold"]
    assert isinstance(t, (int, float)), "threshold must be numeric"
    assert 0.0 < t < 1.0, f"threshold={t} is outside (0, 1)"


def test_metadata_risk_bands_complete(metadata):
    bands = metadata["risk_bands"]
    assert set(bands.keys()) == {"low", "medium", "high"}


def test_metadata_calibration_method(metadata):
    assert metadata["calibration_method"] in ("isotonic", "sigmoid")


def test_metadata_brier_calibrated_better(metadata):
    assert metadata["brier_calibrated"] <= metadata["brier_uncalibrated"], (
        "Calibrated Brier score should be ≤ uncalibrated"
    )


def test_metadata_test_metrics_complete(metadata):
    missing = REQUIRED_METRIC_KEYS - set(metadata["test_metrics"].keys())
    assert not missing, f"Missing test metric keys: {missing}"


def test_metadata_auc_reasonable(metadata):
    auc = metadata["test_metrics"]["roc_auc"]
    assert auc > 0.90, f"Test AUC={auc:.4f} — below expected floor of 0.90"


def test_metadata_n_train_rows(metadata):
    assert metadata["n_train_rows"] == 21742


# ── prediction tests ──────────────────────────────────────────────────────────

def test_predict_proba_shape(model, sample_df):
    proba = model.predict_proba(sample_df)
    assert proba.shape == (1, 2), f"Expected shape (1,2), got {proba.shape}"


def test_predict_proba_sums_to_one(model, sample_df):
    proba = model.predict_proba(sample_df)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_cancellation_probability_in_range(model, sample_df):
    prob = model.predict_proba(sample_df)[0, 1]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"


def test_low_risk_booking(model, metadata):
    """A same-day repeat guest with many special requests should be low risk."""
    low_risk = pd.DataFrame([{
        **SAMPLE_BOOKING,
        "lead_time": 0,
        "repeated_guest": 1,
        "no_of_special_requests": 4,
        "market_segment_type": "Corporate",
        "no_of_previous_bookings_not_canceled": 5,
    }])
    prob = model.predict_proba(low_risk)[0, 1]
    threshold = metadata["threshold"]
    assert prob < threshold, (
        f"Low-risk booking gave probability {prob:.3f} ≥ threshold {threshold:.2f}"
    )


def test_high_risk_booking(model, metadata):
    """A far-future online booking with no requests should be higher risk."""
    high_risk = pd.DataFrame([{
        **SAMPLE_BOOKING,
        "lead_time": 300,
        "repeated_guest": 0,
        "no_of_special_requests": 0,
        "market_segment_type": "Online",
        "no_of_previous_cancellations": 3,
        "avg_price_per_room": 200.0,
        "required_car_parking_space": 0,
    }])
    prob = model.predict_proba(high_risk)[0, 1]
    assert prob > 0.5, (
        f"High-risk booking gave probability {prob:.3f} — expected > 0.5"
    )


def test_risk_band_assignment(model, metadata, sample_df):
    """Verify risk band logic matches metadata thresholds."""
    prob = model.predict_proba(sample_df)[0, 1]
    low_upper  = metadata["risk_bands"]["low"]["upper_exclusive"]
    high_lower = metadata["risk_bands"]["high"]["lower_inclusive"]

    if prob < low_upper:
        band = "low"
    elif prob < high_lower:
        band = "medium"
    else:
        band = "high"

    assert band in ("low", "medium", "high")


def test_batch_prediction(model, sample_df):
    """Pipeline must handle a batch of identical rows."""
    batch = pd.concat([sample_df] * 10, ignore_index=True)
    proba = model.predict_proba(batch)
    assert proba.shape == (10, 2)
    assert np.all(proba[:, 1] >= 0) and np.all(proba[:, 1] <= 1)
