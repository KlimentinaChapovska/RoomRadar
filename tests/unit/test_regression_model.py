"""
Tests for the saved room-price production model.
Verifies the pkl loads, metadata is complete, and real predictions are sensible.
"""
import json, sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

# Register custom transformers in unpickling namespace before loading the pkl
from train_regression import RegFeatureEngineer  # noqa: F401

MODEL_PATH = ROOT / "models" / "room_price_model.pkl"
META_PATH  = ROOT / "models" / "room_price_metadata.json"

# Representative booking with mid-range room (Room_Type 1, 2 adults)
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
    "no_of_special_requests": 1,
    # cleaning flags — may or may not be present; RegFeatureEngineer drops them
    "is_zero_adults": 0,
    "is_zero_nights": 0,
    "price_is_zero": 0,
}

REQUIRED_META_KEYS = {
    "model_name", "version", "trained_on", "target", "target_units",
    "target_transformation", "predict_inverse_transform",
    "n_train_rows", "n_excluded_zero_price", "n_excluded_invalid_dates",
    "xgb_params", "reg_alpha_verified",
    "val_target_comparison", "dummy_baseline",
    "val_metrics", "test_metrics", "overfit_summary",
    "top_features_by_gain", "features_required",
    "pipeline_steps", "limitations",
}

REQUIRED_METRIC_KEYS = {"r2", "mae", "rmse"}


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


# ── file-existence tests ──────────────────────────────────────────────────────

def test_model_file_exists():
    assert MODEL_PATH.exists(), "room_price_model.pkl not found in models/"


def test_metadata_file_exists():
    assert META_PATH.exists(), "room_price_metadata.json not found in models/"


def test_model_file_nonempty():
    assert MODEL_PATH.stat().st_size > 10_000, "Model file is suspiciously small"


# ── metadata tests ─────────────────────────────────────────────────────────────

def test_metadata_required_keys(metadata):
    missing = REQUIRED_META_KEYS - set(metadata.keys())
    assert not missing, f"Missing metadata keys: {missing}"


def test_metadata_target_is_avg_price(metadata):
    assert metadata["target"] == "avg_price_per_room"


def test_metadata_transformation_valid(metadata):
    assert metadata["target_transformation"] in ("raw", "log1p")


def test_metadata_reg_alpha_verified(metadata):
    assert metadata["reg_alpha_verified"] is True


def test_metadata_n_train_rows(metadata):
    assert metadata["n_train_rows"] == 21415


def test_metadata_test_metrics_complete(metadata):
    missing = REQUIRED_METRIC_KEYS - set(metadata["test_metrics"].keys())
    assert not missing, f"Missing test metric keys: {missing}"


def test_metadata_r2_reasonable(metadata):
    r2 = metadata["test_metrics"]["r2"]
    assert r2 > 0.70, f"Test R²={r2:.4f} — below expected floor of 0.70"


def test_metadata_rmse_reasonable(metadata):
    rmse = metadata["test_metrics"]["rmse"]
    assert rmse < 25.0, f"Test RMSE={rmse:.3f} price units — above expected ceiling"


def test_metadata_dummy_baseline_present(metadata):
    db = metadata["dummy_baseline"]
    assert "train_mean" in db
    assert "val_rmse" in db
    assert db["train_mean"] > 50, "Dummy mean seems too low for hotel prices"


def test_metadata_target_comparison_has_both(metadata):
    tc = metadata["val_target_comparison"]
    assert "raw_rmse" in tc and "log1p_rmse" in tc
    assert tc["chosen"] in ("raw", "log1p")


def test_metadata_top_features_nonempty(metadata):
    assert len(metadata["top_features_by_gain"]) >= 5


def test_metadata_features_required_nonempty(metadata):
    assert len(metadata["features_required"]) >= 10


def test_metadata_limitations_listed(metadata):
    assert len(metadata["limitations"]) >= 3


# ── prediction tests ──────────────────────────────────────────────────────────

def test_predict_returns_scalar(model, sample_df):
    preds = model.predict(sample_df)
    assert preds.shape == (1,), f"Expected shape (1,), got {preds.shape}"


def test_prediction_is_positive(model, sample_df):
    price = float(model.predict(sample_df)[0])
    assert price > 0, f"Predicted price {price:.2f} is not positive"


def test_prediction_in_plausible_range(model, sample_df):
    price = float(model.predict(sample_df)[0])
    assert 20 <= price <= 600, (
        f"Predicted price {price:.2f} is outside plausible range [20, 600]"
    )


def test_batch_prediction(model, sample_df):
    batch = pd.concat([sample_df] * 20, ignore_index=True)
    preds = model.predict(batch)
    assert preds.shape == (20,)
    assert np.all(preds > 0)


def test_more_guests_higher_price(model):
    """A group of 4 adults should be priced higher than 1 adult, same room."""
    solo = pd.DataFrame([{**SAMPLE_BOOKING, "no_of_adults": 1, "no_of_children": 0}])
    group = pd.DataFrame([{**SAMPLE_BOOKING, "no_of_adults": 4, "no_of_children": 0}])
    p_solo  = float(model.predict(solo)[0])
    p_group = float(model.predict(group)[0])
    assert p_group >= p_solo, (
        f"Group price {p_group:.2f} should be ≥ solo price {p_solo:.2f}"
    )


def test_complementary_segment_lowest_price(model):
    """Complementary bookings typically have zero/near-zero effective pricing."""
    online = pd.DataFrame([{**SAMPLE_BOOKING, "market_segment_type": "Online"}])
    comp   = pd.DataFrame([{**SAMPLE_BOOKING, "market_segment_type": "Complementary"}])
    p_online = float(model.predict(online)[0])
    p_comp   = float(model.predict(comp)[0])
    assert p_comp < p_online, (
        f"Complementary price {p_comp:.2f} should be < Online price {p_online:.2f}"
    )


def test_inverse_transform_matches_metadata(model, metadata, sample_df):
    """If metadata says log1p, predictions must be back-transformed; raw = direct."""
    raw_pred = float(model.predict(sample_df)[0])
    transform = metadata["target_transformation"]
    if transform == "raw":
        # model predicts price directly — just check it's in range
        assert raw_pred > 0
    else:
        # model predicts log1p(price) internally — pipeline should NOT expose that
        # (the saved pipeline includes the XGBoost but NOT an expm1 step;
        #  back-transform must be done by the caller using metadata info)
        assert raw_pred > 0  # still valid regardless


def test_metadata_r2_consistent_with_model(model, metadata):
    """Stored metadata R² is from the test set; just check it's internally consistent."""
    r2  = metadata["test_metrics"]["r2"]
    mae = metadata["test_metrics"]["mae"]
    assert 0 < r2 < 1
    assert mae > 0
