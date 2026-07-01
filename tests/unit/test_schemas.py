"""Unit tests for Pydantic BookingInput and PredictionResponse schemas."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from app.schemas.booking import BookingInput, ModelVersions, PredictionResponse, PriceComparison

VALID = dict(
    no_of_adults=2,
    no_of_children=0,
    no_of_weekend_nights=1,
    no_of_week_nights=2,
    type_of_meal_plan="Meal Plan 1",
    required_car_parking_space=0,
    room_type_reserved="Room_Type 1",
    lead_time=30,
    arrival_year=2018,
    arrival_month=6,
    arrival_date=15,
    market_segment_type="Online",
    repeated_guest=0,
    no_of_previous_cancellations=0,
    no_of_previous_bookings_not_canceled=0,
    current_room_price=99.0,
    no_of_special_requests=1,
)


def test_valid_booking_parses():
    b = BookingInput(**VALID)
    assert b.no_of_adults == 2
    assert b.current_room_price == 99.0


def test_zero_adults_rejected():
    with pytest.raises(ValidationError):
        BookingInput(**{**VALID, "no_of_adults": 0})


def test_negative_lead_time_rejected():
    with pytest.raises(ValidationError):
        BookingInput(**{**VALID, "lead_time": -1})


def test_invalid_meal_plan_rejected():
    with pytest.raises(ValidationError):
        BookingInput(**{**VALID, "type_of_meal_plan": "Breakfast Only"})


def test_negative_current_room_price_rejected():
    with pytest.raises(ValidationError):
        BookingInput(**{**VALID, "current_room_price": -10.0})


def test_prediction_response_schema():
    comp = PriceComparison(
        current_room_price=99.0,
        predicted_room_price=115.5,
        difference=16.5,
        difference_pct=16.7,
        note="Predicted price is 17% above the current room price — consider reviewing the rate.",
    )
    r = PredictionResponse(
        request_id="abc12345",
        cancellation_probability=0.72,
        cancellation_label="Canceled",
        cancellation_risk_band="high",
        predicted_room_price=115.50,
        price_comparison=comp,
        recommendations=["High risk — request deposit or offer incentive"],
        model_versions=ModelVersions(cancellation="1.0.0", price="1.0.0"),
        prediction_time_ms=42.3,
    )
    assert r.cancellation_label == "Canceled"
    assert r.cancellation_risk_band == "high"
    assert r.price_comparison.current_room_price == 99.0
    assert r.price_comparison.difference == 16.5


def test_booking_schema_uses_current_room_price():
    """current_room_price replaces the old avg_price_per_room field."""
    assert "current_room_price" in BookingInput.model_fields
    assert "avg_price_per_room" not in BookingInput.model_fields


def test_booking_schema_no_current_offered_price():
    """current_offered_price is no longer a separate field."""
    assert "current_offered_price" not in BookingInput.model_fields
