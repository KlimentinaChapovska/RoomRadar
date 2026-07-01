"""
Shared calibration wrapper.  Must live in a stable importable module so that
joblib can deserialise the class regardless of which __main__ is running.
"""
import numpy as np


class CalibratedPipeline:
    """
    Wraps a fitted sklearn Pipeline with a post-hoc probability calibrator.
    Fully picklable; behaves like a sklearn estimator for predict / predict_proba.
    """
    def __init__(self, base_pipeline, calibrator, method: str):
        self.base_pipeline = base_pipeline
        self.calibrator    = calibrator
        self.method        = method

    def _raw_proba(self, X) -> np.ndarray:
        return self.base_pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        raw = self._raw_proba(X)
        if self.method == "isotonic":
            cal = self.calibrator.predict(raw)
        else:                                    # sigmoid / Platt
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
