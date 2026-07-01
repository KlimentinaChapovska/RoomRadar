"""
Shared sklearn transformers for the RoomRadar production pipeline.

Importing this module requires only numpy, pandas, and scikit-learn — no
matplotlib, seaborn, or scipy. model_store.py imports from here so the API
server doesn't pull in heavy plotting dependencies at startup.

The training scripts (train_classification.py, train_regression.py) also
import from here to keep a single source of truth for these classes.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic feature derivation — safe to use inside a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["total_nights"]        = X["no_of_weekend_nights"] + X["no_of_week_nights"]
        X["total_guests"]        = X["no_of_adults"] + X["no_of_children"]
        X["has_children"]        = (X["no_of_children"] > 0).astype(int)
        X["log_lead_time"]       = np.log1p(X["lead_time"])
        X["log_price"]           = np.log1p(X["avg_price_per_room"])
        X["price_is_zero"]       = (X["avg_price_per_room"] == 0).astype(int)
        X["is_weekend_stay"]     = (X["no_of_weekend_nights"] > 0).astype(int)
        X["has_special_request"] = (X["no_of_special_requests"] > 0).astype(int)
        X["prior_cancel_rate"]   = (
            X["no_of_previous_cancellations"]
            / (X["no_of_previous_cancellations"]
               + X["no_of_previous_bookings_not_canceled"] + 1e-9)
        )

        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                      6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        X["arrival_season"] = X["arrival_month"].map(season_map).fillna(0).astype(int)

        X["arrival_dow"] = (
            pd.to_datetime(
                dict(year=X["arrival_year"],
                     month=X["arrival_month"],
                     day=X["arrival_date"]),
                errors="coerce",
            )
            .dt.dayofweek
            .fillna(-1)
            .astype(int)
        )

        drop = ["Booking_ID", "arrival_year", "arrival_date",
                "booking_status", "label",
                "is_zero_adults", "is_zero_nights"]
        return X.drop(columns=[c for c in drop if c in X.columns])


class RegFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature derivation for the price regression pipeline.
    Identical to FeatureEngineer except it does NOT compute log_price or
    price_is_zero (avg_price_per_room is the target).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["total_nights"]        = X["no_of_weekend_nights"] + X["no_of_week_nights"]
        X["total_guests"]        = X["no_of_adults"] + X["no_of_children"]
        X["has_children"]        = (X["no_of_children"] > 0).astype(int)
        X["log_lead_time"]       = np.log1p(X["lead_time"])
        X["is_weekend_stay"]     = (X["no_of_weekend_nights"] > 0).astype(int)
        X["has_special_request"] = (X["no_of_special_requests"] > 0).astype(int)
        X["prior_cancel_rate"]   = (
            X["no_of_previous_cancellations"]
            / (X["no_of_previous_cancellations"]
               + X["no_of_previous_bookings_not_canceled"] + 1e-9)
        )

        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                      6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        X["arrival_season"] = (
            X["arrival_month"].map(season_map).fillna(0).astype(int)
        )
        X["arrival_dow"] = (
            pd.to_datetime(
                dict(year=X["arrival_year"], month=X["arrival_month"],
                     day=X["arrival_date"]),
                errors="coerce",
            )
            .dt.dayofweek.fillna(-1).astype(int)
        )

        drop = [
            "Booking_ID", "arrival_year", "arrival_date",
            "booking_status", "label",
            "avg_price_per_room",
            "is_zero_adults", "is_zero_nights", "price_is_zero",
        ]
        return X.drop(columns=[c for c in drop if c in X.columns])
