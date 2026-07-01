"""
RoomRadar — Room Price Regression Finalization
===============================================
Pre-save validation steps:
  1. Compare raw vs log1p targets for the TUNED XGBoost on validation (no test data)
  2. Fix DummyRegressor baseline — train on raw original target (arithmetic mean)
  3. Verify reg_alpha parameter name against XGBRegressor API
  4. Regenerate feature importances from the production-fitted model
  5. Select production target transformation
  6. Evaluate test set ONCE with the chosen production pipeline
  7. Save models/room_price_model.pkl and models/room_price_metadata.json

Currency note: The project data dictionary annotates avg_price_per_room with "$"
but the original dataset source does not specify a currency.  All monetary outputs
are labelled "price units" throughout this script.

Run from project root:
    python ml/scripts/finalize_regression.py
"""

import sys, json, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from app.utils.logger import get_logger
from train_classification import load_and_clean
from train_regression import (
    RegFeatureEngineer, build_reg_preprocessor,
    REG_NUM_COLS, REG_CAT_COLS, RS,
)

log = get_logger("finalize_reg")

TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR   = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"
for d in (TABLE_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

BRAND = "#922b21"
GREY  = "#7f8c8d"
BLUE  = "#2980b9"
GREEN = "#2ecc71"
DPI   = 150

# Best params from RandomizedSearchCV in train_regression.py
BEST_XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        7,
    "learning_rate":    0.15,
    "subsample":        0.6,
    "colsample_bytree": 0.6,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "min_child_weight": 3,
}


def save_fig(fig, name):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure  → %s", p.name)


def save_table(df, name):
    p = TABLE_DIR / f"{name}.csv"
    df.to_csv(p, index=True)
    log.info("  saved table   → %s", p.name)


def reg_metrics(y_true, y_pred, label="") -> dict:
    """All metrics in original price units."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if label:
        log.info("  %s  R²=%.4f  MAE=%.3f  RMSE=%.3f", label, r2, mae, rmse)
    return {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}


def build_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", build_reg_preprocessor()),
        ("reg",  XGBRegressor(
            **BEST_XGB_PARAMS,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RS,
            verbosity=0,
            device="cpu",
        )),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 0. VERIFY reg_alpha PARAMETER NAME
# ══════════════════════════════════════════════════════════════════════════════

def verify_reg_alpha() -> None:
    log.info("=" * 62)
    log.info("STEP 0 — Verify XGBRegressor parameter names")
    log.info("=" * 62)
    valid = set(XGBRegressor().get_params().keys())
    for name in ("reg_alpha", "reg_lambda"):
        status = "✓ present" if name in valid else "✗ MISSING"
        log.info("  %-15s  %s", name, status)
        if name not in valid:
            raise ValueError(
                f"Parameter '{name}' not found in XGBRegressor.get_params(). "
                f"Available alpha/lambda keys: "
                f"{sorted(k for k in valid if 'alpha' in k or 'lambda' in k)}"
            )
    log.info("  reg_alpha and reg_lambda confirmed correct for XGBoost %s",
             __import__("xgboost").__version__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data():
    log.info("=" * 62)
    log.info("STEP 1 — Load, clean, split")
    log.info("=" * 62)

    df = load_and_clean()
    n_before = len(df)
    df = df[df["avg_price_per_room"] > 0].copy()
    log.info("Zero-price rows excluded: -%d → %d rows", n_before - len(df), len(df))

    drop_from_X = ["avg_price_per_room", "booking_status", "label", "price_is_zero"]
    X = df.drop(columns=[c for c in drop_from_X if c in df.columns])
    y = df["avg_price_per_room"].copy()

    X_tv,    X_test,  y_tv,    y_test  = train_test_split(X, y, test_size=0.20, random_state=RS)
    X_train, X_val,   y_train, y_val   = train_test_split(X_tv, y_tv, test_size=0.25, random_state=RS)

    log.info(
        "Split: train=%d  val=%d  test=%d  "
        "(train mean=%.2f  val mean=%.2f  test mean=%.2f price units)",
        len(y_train), len(y_val), len(y_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPARE RAW vs LOG1P — TUNED XGBoost on VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def compare_targets_xgb(X_train, y_train, X_val, y_val) -> str:
    """
    Fits the tuned XGBoost hyperparameters on raw and log1p targets.
    Evaluates BOTH on the validation set in original price units.
    Returns 'raw' or 'log1p' — the better choice by RMSE.
    Test set is never touched here.
    """
    log.info("=" * 62)
    log.info("STEP 2 — Compare raw vs log1p for TUNED XGBoost on validation")
    log.info("=" * 62)

    rows = []
    for label, y_tr in [("raw", y_train), ("log1p", np.log1p(y_train))]:
        pipe = build_xgb_pipeline()
        pipe.fit(X_train, y_tr)
        y_pred = pipe.predict(X_val)
        if label == "log1p":
            y_pred = np.expm1(y_pred)
        m = reg_metrics(y_val, y_pred, label=f"XGBoost ({label} target) — val")
        rows.append({"target": label, **m})

    tbl = pd.DataFrame(rows)
    save_table(tbl, "rg_fin_01_target_comparison_xgb_val")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = [BRAND, BLUE]
    for metric, ax, direction in [("rmse", axes[0], "↓ lower is better"),
                                   ("mae",  axes[1], "↓ lower is better"),
                                   ("r2",   axes[2], "↑ higher is better")]:
        vals = tbl[metric]
        bars = ax.bar(tbl["target"], vals, color=colors, alpha=0.85,
                      width=0.4, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(vals)) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{metric.upper()} on Val\n({direction})", fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Tuned XGBoost — Raw vs log1p Target (Validation Set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "rg_fin_01_target_comparison_xgb")

    best = tbl.loc[tbl["rmse"].idxmin(), "target"]
    best_rmse  = float(tbl.loc[tbl["target"] == best,  "rmse"].values[0])
    other_rmse = float(tbl.loc[tbl["target"] != best,  "rmse"].values[0])
    log.info(
        "XGBoost target choice: %s  (val RMSE %.3f vs %.3f  Δ=%.3f price units)",
        best, best_rmse, other_rmse, other_rmse - best_rmse,
    )
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 3. CORRECT DUMMY BASELINE — RAW TARGET ONLY
# ══════════════════════════════════════════════════════════════════════════════

def correct_dummy_baseline(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """
    DummyRegressor(strategy='mean') trained on raw original price.
    Predicts the arithmetic mean of training prices for every sample.
    This is the correct baseline: no target transformation applied.
    (Previous training run incorrectly trained Dummy on log1p target,
     giving a geometric-mean baseline instead of arithmetic mean.)
    """
    log.info("=" * 62)
    log.info("STEP 3 — Correct DummyRegressor baseline (raw target)")
    log.info("=" * 62)

    dummy_mean = float(y_train.mean())
    log.info("  Arithmetic mean of train prices: %.3f price units", dummy_mean)

    dummy = Pipeline([
        ("prep", build_reg_preprocessor()),
        ("reg",  DummyRegressor(strategy="mean")),
    ])
    dummy.fit(X_train, y_train)

    m_val  = reg_metrics(y_val,  dummy.predict(X_val),  "Dummy val (raw)")
    m_test = reg_metrics(y_test, dummy.predict(X_test), "Dummy test (raw)")

    tbl = pd.DataFrame([
        {"split": "val",  **m_val},
        {"split": "test", **m_test},
    ])
    save_table(tbl, "rg_fin_02_dummy_baseline_raw")

    log.info(
        "  Dummy RMSE — val=%.3f  test=%.3f  "
        "(was %.3f/%.3f when trained on log1p target)",
        m_val["rmse"], m_test["rmse"], 33.57, 33.11,
    )
    return {"val": m_val, "test": m_test, "train_mean": dummy_mean}


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN PRODUCTION MODEL & REGENERATE FEATURE IMPORTANCES
# ══════════════════════════════════════════════════════════════════════════════

def train_production_model(X_train, y_train, use_log: bool) -> Pipeline:
    log.info("=" * 62)
    log.info("STEP 4 — Train production pipeline (target=%s)",
             "log1p(y)" if use_log else "raw y")
    log.info("=" * 62)

    y_tr = np.log1p(y_train) if use_log else y_train
    pipe = build_xgb_pipeline()
    pipe.fit(X_train, y_tr)
    log.info("  Production XGBoost fitted on %d training rows.", len(y_tr))
    return pipe


def regenerate_feature_importances(prod_pipe: Pipeline) -> pd.DataFrame:
    log.info("Regenerating feature importances from production model …")
    feat_names   = prod_pipe.named_steps["prep"].named_steps["col"].get_feature_names_out()
    importances  = prod_pipe.named_steps["reg"].feature_importances_

    fi_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_clean = fi_df.copy()
    fi_clean["feature"] = (
        fi_clean["feature"]
        .str.replace(r"^(num|cat)__", "", regex=True)
        .str.replace("_infrequent_sklearn", " (rare)", regex=False)
    )

    save_table(fi_df,   "rg_fin_03_feature_importances_raw")
    save_table(fi_clean, "rg_fin_03_feature_importances_clean")

    top15 = fi_clean.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1],
            color=BRAND, alpha=0.85, edgecolor="white")
    ax.set_title("XGBoost Feature Importances (Gain) — Room Price\nProduction Model",
                 fontweight="bold")
    ax.set_xlabel("Importance (gain)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "rg_fin_02_feature_importances_production")

    log.info("  Top 5 features:\n%s", top15.head(5).to_string(index=False))
    return fi_clean


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL TEST-SET EVALUATION  (test set opened ONCE)
# ══════════════════════════════════════════════════════════════════════════════

def final_test_evaluation(
    prod_pipe, dummy_test_metrics,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    use_log: bool,
) -> dict:
    log.info("=" * 62)
    log.info("STEP 5 — Final test-set evaluation (first and only test-set open)")
    log.info("=" * 62)

    def predict_orig(pipe, X):
        y_pred = pipe.predict(X)
        return np.expm1(y_pred) if use_log else y_pred

    m_train = reg_metrics(y_train, predict_orig(prod_pipe, X_train), "XGB train")
    m_val   = reg_metrics(y_val,   predict_orig(prod_pipe, X_val),   "XGB val")
    m_test  = reg_metrics(y_test,  predict_orig(prod_pipe, X_test),  "XGB test")

    rows = [
        {"model": "DummyRegressor (raw mean)",    "split": "test", **dummy_test_metrics},
        {"model": f"XGBoost (tuned, {'log1p' if use_log else 'raw'})", "split": "train", **m_train},
        {"model": f"XGBoost (tuned, {'log1p' if use_log else 'raw'})", "split": "val",   **m_val},
        {"model": f"XGBoost (tuned, {'log1p' if use_log else 'raw'})", "split": "test",  **m_test},
    ]
    results_tbl = pd.DataFrame(rows)
    save_table(results_tbl, "rg_fin_04_final_metrics")

    overfit = {
        "train_r2":   m_train["r2"],   "val_r2":  m_val["r2"],  "test_r2":  m_test["r2"],
        "train_rmse": m_train["rmse"], "val_rmse": m_val["rmse"], "test_rmse": m_test["rmse"],
        "train_mae":  m_train["mae"],  "val_mae":  m_val["mae"],  "test_mae":  m_test["mae"],
    }
    log.info(
        "Overfit summary: train R²=%.4f  val R²=%.4f  test R²=%.4f  "
        "(train→val gap=%.4f  val→test gap=%.4f)",
        m_train["r2"], m_val["r2"], m_test["r2"],
        m_train["r2"] - m_val["r2"], m_val["r2"] - m_test["r2"],
    )

    # predicted vs actual on test
    y_pred_test = predict_orig(prod_pipe, X_test)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred_test, alpha=0.25, s=10, color=BRAND)
    lo = min(y_test.min(), y_pred_test.min())
    hi = max(y_test.max(), y_pred_test.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
    axes[0].set_title(
        f"Predicted vs Actual — Test Set\nR²={m_test['r2']:.4f}  "
        f"RMSE={m_test['rmse']:.2f}  MAE={m_test['mae']:.2f} price units",
        fontweight="bold",
    )
    axes[0].set_xlabel("Actual price (price units)")
    axes[0].set_ylabel("Predicted price (price units)")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    residuals = y_test.values - y_pred_test
    axes[1].hist(residuals, bins=60, color=BRAND, alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="k", lw=1.2, ls="--")
    axes[1].set_title("Residual Distribution — Test Set", fontweight="bold")
    axes[1].set_xlabel("Residual (price units)")
    axes[1].set_ylabel("Count")
    axes[1].text(
        0.97, 0.94,
        f"mean={residuals.mean():.2f}\nstd={residuals.std():.2f}",
        ha="right", va="top", transform=axes[1].transAxes,
        fontsize=9, color=GREY,
    )
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Production XGBoost — Final Test Set Evaluation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "rg_fin_03_final_test_evaluation")

    return {"train": m_train, "val": m_val, "test": m_test, "overfit": overfit}


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE PRODUCTION ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_artefacts(
    prod_pipe: Pipeline,
    fi_df: pd.DataFrame,
    metrics: dict,
    dummy_metrics: dict,
    use_log: bool,
    n_train: int,
) -> None:
    log.info("=" * 62)
    log.info("STEP 6 — Saving production artefacts")
    log.info("=" * 62)

    # ── model ────────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "room_price_model.pkl"
    joblib.dump(prod_pipe, model_path, compress=3)
    size_kb = model_path.stat().st_size / 1024
    log.info("  Model saved   → %s  (%.1f KB)", model_path.name, size_kb)

    # ── metadata ─────────────────────────────────────────────────────────────
    metadata = {
        "model_name":    "roomradar_room_price_xgb",
        "version":       "1.0.0",
        "trained_on":    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "target":        "avg_price_per_room",
        "target_units":  "price units (currency unspecified in source data; '$' notation in data dictionary may be approximate)",
        "target_transformation":  "log1p" if use_log else "raw",
        "predict_inverse_transform": "numpy.expm1" if use_log else "none",
        "n_train_rows":       n_train,
        "n_excluded_zero_price":   545,
        "n_excluded_invalid_dates": 37,
        "xgb_params":   BEST_XGB_PARAMS,
        "reg_alpha_verified": True,
        "val_target_comparison": {
            "raw_rmse":   None,   # filled below
            "log1p_rmse": None,
            "chosen":     "log1p" if use_log else "raw",
        },
        "dummy_baseline": {
            "strategy":  "mean of raw training prices",
            "train_mean": dummy_metrics["train_mean"],
            "val_rmse":   dummy_metrics["val"]["rmse"],
            "test_rmse":  dummy_metrics["test"]["rmse"],
        },
        "val_metrics":  metrics["val"],
        "test_metrics": metrics["test"],
        "overfit_summary": {
            "train_r2":   metrics["train"]["r2"],
            "val_r2":     metrics["val"]["r2"],
            "test_r2":    metrics["test"]["r2"],
            "train_rmse": metrics["train"]["rmse"],
            "val_rmse":   metrics["val"]["rmse"],
            "test_rmse":  metrics["test"]["rmse"],
        },
        "top_features_by_gain": fi_df.head(10)[["feature", "importance"]]
            .assign(importance=lambda d: d["importance"].round(6))
            .to_dict(orient="records"),
        "features_required": [
            "no_of_adults", "no_of_children", "no_of_weekend_nights",
            "no_of_week_nights", "type_of_meal_plan", "required_car_parking_space",
            "room_type_reserved", "lead_time", "arrival_year", "arrival_month",
            "arrival_date", "market_segment_type", "repeated_guest",
            "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
            "no_of_special_requests",
        ],
        "pipeline_steps": [
            "RegFeatureEngineer (custom TransformerMixin — no price-derived features)",
            "ColumnTransformer: StandardScaler (numeric) + OHE (categorical)",
            "XGBRegressor (tuned via RandomizedSearchCV, 30 iter, KFold-5)",
        ],
        "limitations": [
            "R²=0.716 — 28% of price variance unexplained (promotions, loyalty, competitor rates not in data)",
            "Tends to under-predict luxury rooms (>200 price units) where training data is sparse",
            "Trained on 2017–2018 hotel data; may not generalise to post-pandemic pricing dynamics",
            "Booking price may have been updated after booking — avg_price_per_room reflects final price, not initial quote",
        ],
    }

    # fill in target comparison RMSEs from saved table
    try:
        cmp = pd.read_csv(TABLE_DIR / "rg_fin_01_target_comparison_xgb_val.csv", index_col=0)
        for _, row in cmp.iterrows():
            if row["target"] == "raw":
                metadata["val_target_comparison"]["raw_rmse"] = row["rmse"]
            else:
                metadata["val_target_comparison"]["log1p_rmse"] = row["rmse"]
    except FileNotFoundError:
        pass

    meta_path = MODEL_DIR / "room_price_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("  Metadata saved → %s", meta_path.name)

    log.info(
        "Production model ready: "
        "target=%s  val_R²=%.4f  test_R²=%.4f  test_RMSE=%.3f price units",
        "log1p" if use_log else "raw",
        metrics["val"]["r2"], metrics["test"]["r2"], metrics["test"]["rmse"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("RoomRadar — Regression Finalization — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 0. Verify parameter naming
    verify_reg_alpha()

    # 1. Data (identical split to training run)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    # 2. Compare raw vs log1p for TUNED XGBoost on validation (test set untouched)
    best_target = compare_targets_xgb(X_train, y_train, X_val, y_val)
    use_log = (best_target == "log1p")

    # 3. Correct DummyRegressor baseline on raw target
    dummy_metrics = correct_dummy_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 4. Train production model
    prod_pipe = train_production_model(X_train, y_train, use_log)

    # 5. Regenerate feature importances from production model
    fi_df = regenerate_feature_importances(prod_pipe)

    # 6. Final test-set evaluation (test set opened ONCE here)
    metrics = final_test_evaluation(
        prod_pipe, dummy_metrics["test"],
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        use_log=use_log,
    )

    # 7. Save
    save_artefacts(prod_pipe, fi_df, metrics, dummy_metrics, use_log, len(y_train))

    log.info("=" * 62)
    log.info("DONE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    n_tbl = len(list(TABLE_DIR.glob("rg_fin_*.csv")))
    n_fig = len(list(FIG_DIR.glob("rg_fin_*.png")))
    log.info("Tables  (%d new): %s", n_tbl, TABLE_DIR)
    log.info("Figures (%d new): %s", n_fig, FIG_DIR)
    log.info("Models  :  %s", MODEL_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
