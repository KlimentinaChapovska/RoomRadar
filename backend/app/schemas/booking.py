"""Pydantic schemas for booking input and prediction response."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class BookingInput(BaseModel):
    no_of_adults: int = Field(..., ge=1, le=4)
    no_of_children: int = Field(0, ge=0, le=10)
    no_of_weekend_nights: int = Field(..., ge=0, le=7)
    no_of_week_nights: int = Field(..., ge=0, le=17)
    type_of_meal_plan: Literal["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
    required_car_parking_space: Literal[0, 1]
    room_type_reserved: Literal[
        "Room_Type 1", "Room_Type 2", "Room_Type 3",
        "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"
    ]
    lead_time: int = Field(..., ge=0)
    arrival_year: int = Field(..., ge=2017)
    arrival_month: int = Field(..., ge=1, le=12)
    arrival_date: int = Field(..., ge=1, le=31)
    market_segment_type: Literal["Online", "Offline", "Corporate", "Complementary", "Aviation"]
    repeated_guest: Literal[0, 1]
    no_of_previous_cancellations: int = Field(0, ge=0)
    no_of_previous_bookings_not_canceled: int = Field(0, ge=0)
    current_room_price: float = Field(
        ..., ge=0,
        description=(
            "Current room price in price units. "
            "Mapped to avg_price_per_room for the cancellation model; "
            "compared against the regression model's predicted price."
        ),
    )
    no_of_special_requests: int = Field(0, ge=0, le=5)

    @field_validator("no_of_adults")
    @classmethod
    def adults_not_zero(cls, v: int) -> int:
        if v == 0:
            raise ValueError("Booking must have at least 1 adult.")
        return v


class PriceComparison(BaseModel):
    current_room_price: float = Field(..., description="Current price from the request (price units)")
    predicted_room_price: float = Field(..., description="Model's predicted price (price units)")
    difference: float = Field(..., description="predicted − current (price units)")
    difference_pct: float = Field(..., description="Percentage difference (predicted vs current)")
    note: str


class ModelVersions(BaseModel):
    cancellation: str
    price: str


class PredictionResponse(BaseModel):
    request_id: str
    cancellation_probability: float = Field(..., ge=0.0, le=1.0)
    cancellation_label: Literal["Canceled", "Not_Canceled"]
    cancellation_risk_band: Literal["low", "medium", "high"]
    predicted_room_price: float = Field(..., ge=0.0, description="Predicted price in price units")
    price_comparison: PriceComparison
    recommendations: List[str]
    model_versions: ModelVersions
    prediction_time_ms: float
