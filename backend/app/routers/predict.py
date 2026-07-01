"""Prediction endpoints — cancellation probability and room price."""
import time
import uuid
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models import model_store
from app.schemas.booking import (
    BookingInput,
    ModelVersions,
    PredictionResponse,
    PriceComparison,
)
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["predict"])


# ── input preparation ─────────────────────────────────────────────────────────

def _build_clf_df(b: BookingInput) -> pd.DataFrame:
    """DataFrame for the cancellation pipeline.

    current_room_price maps to the avg_price_per_room column that the
    cancellation model was trained on.  current_room_price is NOT passed
    to the regression model.
    """
    return pd.DataFrame([{
        "no_of_adults":                         b.no_of_adults,
        "no_of_children":                       b.no_of_children,
        "no_of_weekend_nights":                 b.no_of_weekend_nights,
        "no_of_week_nights":                    b.no_of_week_nights,
        "type_of_meal_plan":                    b.type_of_meal_plan,
        "required_car_parking_space":           b.required_car_parking_space,
        "room_type_reserved":                   b.room_type_reserved,
        "lead_time":                            b.lead_time,
        "arrival_year":                         b.arrival_year,
        "arrival_month":                        b.arrival_month,
        "arrival_date":                         b.arrival_date,
        "market_segment_type":                  b.market_segment_type,
        "repeated_guest":                       b.repeated_guest,
        "no_of_previous_cancellations":         b.no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": b.no_of_previous_bookings_not_canceled,
        "avg_price_per_room":                   b.current_room_price,   # API name → model column name
        "no_of_special_requests":               b.no_of_special_requests,
        # cleaning flags — FeatureEngineer drops these; safe to include
        "is_zero_adults": int(b.no_of_adults == 0),
        "is_zero_nights": int(b.no_of_weekend_nights + b.no_of_week_nights == 0),
    }])


def _build_reg_df(b: BookingInput) -> pd.DataFrame:
    """DataFrame for the price regression pipeline.

    avg_price_per_room / current_room_price are intentionally absent —
    avg_price_per_room is the regression target, not a predictor.
    """
    return pd.DataFrame([{
        "no_of_adults":                         b.no_of_adults,
        "no_of_children":                       b.no_of_children,
        "no_of_weekend_nights":                 b.no_of_weekend_nights,
        "no_of_week_nights":                    b.no_of_week_nights,
        "type_of_meal_plan":                    b.type_of_meal_plan,
        "required_car_parking_space":           b.required_car_parking_space,
        "room_type_reserved":                   b.room_type_reserved,
        "lead_time":                            b.lead_time,
        "arrival_year":                         b.arrival_year,
        "arrival_month":                        b.arrival_month,
        "arrival_date":                         b.arrival_date,
        "market_segment_type":                  b.market_segment_type,
        "repeated_guest":                       b.repeated_guest,
        "no_of_previous_cancellations":         b.no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": b.no_of_previous_bookings_not_canceled,
        "no_of_special_requests":               b.no_of_special_requests,
        # cleaning flags — RegFeatureEngineer drops these; safe to include
        "is_zero_adults": int(b.no_of_adults == 0),
        "is_zero_nights": int(b.no_of_weekend_nights + b.no_of_week_nights == 0),
    }])


# ── business logic ────────────────────────────────────────────────────────────

def _get_risk_band(prob: float, risk_bands: dict) -> str:
    low_upper  = risk_bands["low"]["upper_exclusive"]
    high_lower = risk_bands["high"]["lower_inclusive"]
    if prob < low_upper:
        return "low"
    elif prob < high_lower:
        return "medium"
    else:
        return "high"


def _build_price_comparison(current: float, predicted: float) -> PriceComparison:
    diff = predicted - current
    pct  = (diff / current * 100) if current > 0 else 0.0
    if abs(pct) < 5:
        note = "Predicted price closely matches the current room price."
    elif pct > 0:
        note = (
            f"Predicted price is {pct:.0f}% above the current room price "
            "— consider reviewing the rate."
        )
    else:
        note = (
            f"Current room price is {abs(pct):.0f}% above the predicted rate "
            "— strong value for the guest."
        )
    return PriceComparison(
        current_room_price=round(current, 2),
        predicted_room_price=round(predicted, 2),
        difference=round(diff, 2),
        difference_pct=round(pct, 1),
        note=note,
    )


def _build_recommendations(
    risk_band: str,
    clf_meta: dict,
    lead_time: int,
    no_of_special_requests: int,
    price_comparison: PriceComparison,
) -> List[str]:
    recs: list[str] = []

    # Primary: risk band description from metadata
    recs.append(clf_meta["risk_bands"][risk_band]["description"])

    # Price gap insight — only when the gap is meaningful (≥ 10 %)
    if abs(price_comparison.difference_pct) >= 10:
        recs.append(price_comparison.note)

    # Long lead time + elevated risk → proactive outreach
    if lead_time > 150 and risk_band in ("medium", "high"):
        recs.append(
            "Long lead time combined with elevated cancellation risk "
            "— consider early guest outreach."
        )

    # No special requests + high risk → guest may be disengaged
    if no_of_special_requests == 0 and risk_band == "high":
        recs.append(
            "No special requests on file "
            "— confirm booking details directly with the guest."
        )

    return recs


# ── endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict cancellation probability and room price",
    responses={
        503: {"description": "Models not loaded."},
        422: {"description": "Invalid booking input."},
        500: {"description": "Unexpected server error."},
    },
)
async def predict_booking(booking: BookingInput):
    """
    Returns:

    - **cancellation_probability**: calibrated float 0–1
    - **cancellation_label**: Canceled | Not_Canceled (threshold from metadata)
    - **cancellation_risk_band**: low | medium | high
    - **predicted_room_price**: float in price units
    - **price_comparison**: predicted price vs current_room_price
    - **recommendations**: rule-based business actions
    """
    request_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()

    log.info(
        "Prediction request [%s] room=%s market=%s lead_time=%d",
        request_id,
        booking.room_type_reserved,
        booking.market_segment_type,
        booking.lead_time,
    )

    if not model_store.is_ready():
        log.warning("Prediction [%s] rejected — models not loaded.", request_id)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "models_not_ready",
                "message": (
                    "Prediction models are not loaded. "
                    "Run ml/scripts/finalize_classification.py and "
                    "ml/scripts/finalize_regression.py, then restart the server."
                ),
            },
        )

    try:
        clf      = model_store.get_cancellation_model()
        clf_meta = model_store.get_cancellation_metadata()
        reg      = model_store.get_price_model()
        reg_meta = model_store.get_price_metadata()

        # Cancellation — current_room_price is mapped to avg_price_per_room inside the DataFrame
        clf_df      = _build_clf_df(booking)
        cancel_prob = float(clf.predict_proba(clf_df)[0, 1])
        threshold   = float(clf_meta["threshold"])
        cancel_label = "Canceled" if cancel_prob >= threshold else "Not_Canceled"
        risk_band    = _get_risk_band(cancel_prob, clf_meta["risk_bands"])

        # Price — current_room_price is not passed to this model at all
        reg_df          = _build_reg_df(booking)
        predicted_price = float(reg.predict(reg_df)[0])
        if reg_meta.get("target_transformation") == "log1p":
            predicted_price = float(np.expm1(predicted_price))
        predicted_price = max(predicted_price, 0.0)

        # Price comparison — always computed from current_room_price
        price_comp = _build_price_comparison(booking.current_room_price, predicted_price)

        # Recommendations
        recommendations = _build_recommendations(
            risk_band, clf_meta,
            booking.lead_time, booking.no_of_special_requests, price_comp,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Prediction [%s] band=%s prob=%.4f price=%.2f  %.1fms",
            request_id, risk_band, cancel_prob, predicted_price, elapsed_ms,
        )

        return PredictionResponse(
            request_id=request_id,
            cancellation_probability=round(cancel_prob, 4),
            cancellation_label=cancel_label,
            cancellation_risk_band=risk_band,
            predicted_room_price=round(predicted_price, 2),
            price_comparison=price_comp,
            recommendations=recommendations,
            model_versions=ModelVersions(
                cancellation=clf_meta.get("version", "unknown"),
                price=reg_meta.get("version", "unknown"),
            ),
            prediction_time_ms=round(elapsed_ms, 1),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        log.error("Value error in prediction [%s]: %s", request_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.error("Unexpected error in prediction [%s]: %s", request_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "request_id": request_id,
                "message": "An unexpected error occurred. Please try again.",
            },
        )
