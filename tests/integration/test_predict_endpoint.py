"""Integration tests for /api/v1/predict and /api/v1/model-info endpoints."""
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.models.model_store as ms
from app.main import app


# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def load_models_once():
    """Load models once for all tests in this module."""
    ms.load_models()
    assert ms.is_ready(), "Models failed to load — check PKL files in models/"


VALID_BOOKING = {
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
    "current_room_price": 99.0,
    "no_of_special_requests": 1,
}


# ── model loading ─────────────────────────────────────────────────────────────

def test_model_store_is_ready():
    assert ms.is_ready() is True


def test_cancellation_model_loaded():
    assert ms.get_cancellation_model() is not None


def test_price_model_loaded():
    assert ms.get_price_model() is not None


def test_cancellation_metadata_has_threshold():
    meta = ms.get_cancellation_metadata()
    assert "threshold" in meta
    assert 0 < meta["threshold"] < 1


def test_price_metadata_has_target():
    meta = ms.get_price_metadata()
    assert meta["target"] == "avg_price_per_room"


# ── /api/v1/model-info ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_info_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/model-info")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_model_info_has_both_models():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/model-info")
    body = r.json()
    assert "cancellation_model" in body
    assert "price_model" in body


@pytest.mark.asyncio
async def test_model_info_cancellation_has_risk_bands():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/model-info")
    clf = r.json()["cancellation_model"]
    assert "risk_bands" in clf
    assert set(clf["risk_bands"].keys()) == {"low", "medium", "high"}


@pytest.mark.asyncio
async def test_model_info_excludes_internal_fields():
    """Hyperparameters, dataset counts, and validation artefacts are not exposed."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/model-info")
    body = r.json()
    clf = body["cancellation_model"]
    reg = body["price_model"]
    for internal in ("xgb_params", "pipeline_steps", "n_train_rows", "overfit_summary"):
        assert internal not in clf, f"Internal field '{internal}' exposed in cancellation_model"
        assert internal not in reg, f"Internal field '{internal}' exposed in price_model"


@pytest.mark.asyncio
async def test_model_info_503_when_not_ready(monkeypatch):
    monkeypatch.setattr(ms, "is_ready", lambda: False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/v1/model-info")
    assert r.status_code == 503


# ── /api/v1/predict — valid input ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_predict_response_has_required_fields():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    body = r.json()
    for field in [
        "request_id", "cancellation_probability", "cancellation_label",
        "cancellation_risk_band", "predicted_room_price",
        "price_comparison", "recommendations", "model_versions", "prediction_time_ms",
    ]:
        assert field in body, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_predict_probability_in_range():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    prob = r.json()["cancellation_probability"]
    assert 0.0 <= prob <= 1.0


@pytest.mark.asyncio
async def test_predict_label_is_valid():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert r.json()["cancellation_label"] in ("Canceled", "Not_Canceled")


@pytest.mark.asyncio
async def test_predict_risk_band_is_valid():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert r.json()["cancellation_risk_band"] in ("low", "medium", "high")


@pytest.mark.asyncio
async def test_predict_room_price_is_positive():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert r.json()["predicted_room_price"] > 0


@pytest.mark.asyncio
async def test_predict_recommendations_nonempty():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert len(r.json()["recommendations"]) >= 1


@pytest.mark.asyncio
async def test_predict_has_model_versions():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    mv = r.json()["model_versions"]
    assert "cancellation" in mv and "price" in mv


@pytest.mark.asyncio
async def test_predict_has_request_id():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert len(r.json()["request_id"]) == 8


@pytest.mark.asyncio
async def test_predict_timing_positive():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert r.json()["prediction_time_ms"] > 0


# ── /api/v1/predict — price_comparison always present ────────────────────────

@pytest.mark.asyncio
async def test_predict_price_comparison_always_present():
    """price_comparison is always populated because current_room_price is required."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    comp = r.json()["price_comparison"]
    assert comp is not None
    assert "current_room_price" in comp
    assert "predicted_room_price" in comp
    assert "difference" in comp
    assert "difference_pct" in comp
    assert "note" in comp


@pytest.mark.asyncio
async def test_predict_comparison_echoes_current_room_price():
    """price_comparison.current_room_price must match the request field."""
    payload = {**VALID_BOOKING, "current_room_price": 120.0}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=payload)
    assert r.json()["price_comparison"]["current_room_price"] == 120.0


@pytest.mark.asyncio
async def test_predict_comparison_note_uses_price_units():
    """Comparison notes must not reference a specific currency."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    note = r.json()["price_comparison"]["note"]
    for currency_symbol in ("$", "€", "£", "USD", "EUR"):
        assert currency_symbol not in note, f"Currency symbol '{currency_symbol}' found in note: {note}"


@pytest.mark.asyncio
async def test_predict_current_room_price_not_in_regression():
    """Different current_room_price values must not change the predicted price.
    The regression model does not receive this field at all."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r_low  = await client.post("/api/v1/predict", json={**VALID_BOOKING, "current_room_price": 50.0})
        r_high = await client.post("/api/v1/predict", json={**VALID_BOOKING, "current_room_price": 500.0})
    assert r_low.json()["predicted_room_price"] == r_high.json()["predicted_room_price"]


# ── /api/v1/predict — invalid input (Pydantic validation) ────────────────────

@pytest.mark.asyncio
async def test_predict_rejects_zero_adults():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json={**VALID_BOOKING, "no_of_adults": 0})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_negative_lead_time():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json={**VALID_BOOKING, "lead_time": -1})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_missing_required_field():
    payload = {k: v for k, v in VALID_BOOKING.items() if k != "room_type_reserved"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_unknown_categorical():
    """Values outside the Literal enum are caught by Pydantic before reaching the model."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json={**VALID_BOOKING, "type_of_meal_plan": "Meal Plan 99"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_invalid_market_segment():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json={**VALID_BOOKING, "market_segment_type": "Unknown"})
    assert r.status_code == 422


# ── /api/v1/predict — models not loaded ──────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_returns_503_when_models_not_loaded(monkeypatch):
    monkeypatch.setattr(ms, "is_ready", lambda: False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/v1/predict", json=VALID_BOOKING)
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["error"] == "models_not_ready"
