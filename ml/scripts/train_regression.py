"""
RoomRadar — Room Price Regression Pipeline
==========================================
Predicts avg_price_per_room for a hotel booking.

Cleaning applied:
  • Invalid calendar dates removed (37 rows)
  • Zero-price rows excluded for regression only (545 rows)
  • Booking_ID, booking_status, avg_price_per_room excluded from predictors

Steps
-----
1. Load & clean
2. Split 60/20/20 (random_state=42, no stratification)
3. Compare raw vs log1p(y) on validation
4. Train DummyRegressor, Ridge, DecisionTree, XGBoost
5. DT overfitting curve
6. XGBoost hyperparameter tuning (RandomizedSearchCV, 30 iter, KFold-5)
7. Test-set evaluation  (opened ONCE at the end)
8. Save tables and figures

Run from project root:
    python ml/scripts/train_regression.py

Does NOT save a production model.
"""

import sys, warnings, calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from app.utils.logger import get_logger
from train_classification import load_and_clean

log = get_logger("train_reg")

RAW_CSV   = ROOT / "data" / "raw" / "Hotel Reservations.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR   = ROOT / "outputs" / "figures"
for d in (TABLE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

BRAND = "#922b21"
GREY  = "#7f8c8d"
BLUE  = "#2980b9"
GREEN = "#2ecc71"
DPI   = 150
RS    = 42

# ── column definitions (regression — excludes target and its derivatives) ────
REG_NUM_COLS = [
    # raw numeric
    "no_of_adults", "no_of_children",
    "no_of_weekend_nights", "no_of_week_nights",
    "lead_time", "arrival_month",
    "required_car_parking_space", "repeated_guest",
    "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
    "no_of_special_requests",
    # engineered
    "total_nights", "total_guests", "has_children",
    "log_lead_time",
    "is_weekend_stay", "arrival_season", "arrival_dow",
    "has_special_request", "prior_cancel_rate",
]
REG_CAT_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]


# ── helpers ──────────────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, name: str) -> None:
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure  → %s", p.name)


def save_table(df: pd.DataFrame, name: str) -> None:
    p = TABLE_DIR / f"{name}.csv"
    df.to_csv(p, index=True)
    log.info("  saved table   → %s", p.name)


def reg_metrics(y_true, y_pred, label: str = "") -> dict:
    """R², MAE, RMSE in original price units."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if label:
        log.info("  %s   R²=%.4f  MAE=%.2f  RMSE=%.2f", label, r2, mae, rmse)
    return {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEER (regression — no price-derived features)
# ══════════════════════════════════════════════════════════════════════════════

class RegFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature derivation for the price regression pipeline.
    Identical to FeatureEngineer except:
      • does NOT compute log_price or price_is_zero  (avg_price_per_room is the TARGET)
      • safe to call whether or not those columns are present in X
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
            "avg_price_per_room",   # target — remove if accidentally present
            "is_zero_adults", "is_zero_nights", "price_is_zero",
        ]
        return X.drop(columns=[c for c in drop if c in X.columns])


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_reg_preprocessor() -> Pipeline:
    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), REG_NUM_COLS),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
                REG_CAT_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("eng", RegFeatureEngineer()), ("col", col_transformer)])


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD, FILTER, SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def prepare_regression_data():
    log.info("=" * 62)
    log.info("STEP 1 — Load, clean, filter zero-price rows")
    log.info("=" * 62)

    df = load_and_clean()                                   # 36,238 after date fix
    n_before = len(df)
    df = df[df["avg_price_per_room"] > 0].copy()            # drop 545 zero-price
    log.info(
        "Zero-price rows excluded for regression: -%d → %d rows remain",
        n_before - len(df), len(df),
    )

    # Drop target-derived and label columns from feature matrix
    drop_from_X = ["avg_price_per_room", "booking_status", "label", "price_is_zero"]
    X = df.drop(columns=[c for c in drop_from_X if c in df.columns])
    y = df["avg_price_per_room"].copy()

    return X, y


def split_regression(X: pd.DataFrame, y: pd.Series):
    log.info("=" * 62)
    log.info("STEP 2 — Train / val / test split (60/20/20, RS=%d)", RS)
    log.info("=" * 62)

    X_tv,  X_test,  y_tv,  y_test  = train_test_split(X, y, test_size=0.20, random_state=RS)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=RS)

    rows = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }
    tbl_rows = []
    for name, (Xs, ys) in rows.items():
        log.info(
            "  %-6s  %6d rows | price  mean=%.2f  median=%.2f  std=%.2f",
            name, len(ys), ys.mean(), ys.median(), ys.std(),
        )
        tbl_rows.append({
            "split":  name,
            "n_rows": len(ys),
            "mean":   round(ys.mean(),   2),
            "median": round(ys.median(), 2),
            "std":    round(ys.std(),    2),
            "min":    round(ys.min(),    2),
            "max":    round(ys.max(),    2),
        })

    save_table(pd.DataFrame(tbl_rows), "rg_01_split_sizes")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAW vs LOG1P TARGET COMPARISON  (validation only)
# ══════════════════════════════════════════════════════════════════════════════

def compare_targets(X_train, y_train, X_val, y_val) -> str:
    """
    Fits a Ridge regression on raw and log1p(y) using the same training set.
    Evaluates RMSE on the validation set in *original price units*.
    Returns 'raw' or 'log' — the better transformation for all subsequent models.
    """
    log.info("=" * 62)
    log.info("STEP 3 — Compare raw vs log1p(y) on validation (no test data)")
    log.info("=" * 62)

    y_train_log = np.log1p(y_train)

    results = []
    for label, y_tr in [("raw", y_train), ("log1p", y_train_log)]:
        pipe = Pipeline([
            ("prep", build_reg_preprocessor()),
            ("reg",  Ridge(alpha=1.0)),
        ])
        pipe.fit(X_train, y_tr)
        y_pred = pipe.predict(X_val)
        if label == "log1p":
            y_pred = np.expm1(y_pred)      # back-transform for fair comparison
        m = reg_metrics(y_val, y_pred, label=f"Ridge ({label} target) on val")
        results.append({"target": label, **m})

    tbl = pd.DataFrame(results)
    save_table(tbl, "rg_02_target_comparison")

    # ── target distribution figure ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(y_train, bins=50, color=BRAND, alpha=0.8, edgecolor="white")
    axes[0].set_title("Raw: avg_price_per_room", fontweight="bold")
    axes[0].set_xlabel("Price (€)")
    axes[0].set_ylabel("Count")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(np.log1p(y_train), bins=50, color=BLUE, alpha=0.8, edgecolor="white")
    axes[1].set_title("log1p(avg_price_per_room)", fontweight="bold")
    axes[1].set_xlabel("log1p(price)")
    axes[1].set_ylabel("Count")
    axes[1].spines[["top", "right"]].set_visible(False)

    for ax, (label, ys) in zip(axes, [("raw", y_train), ("log", np.log1p(y_train))]):
        skew = float(pd.Series(ys).skew())
        ax.text(0.97, 0.95, f"skew={skew:.2f}", ha="right", va="top",
                transform=ax.transAxes, fontsize=9, color=GREY)

    fig.suptitle("Target Distribution: Training Set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "rg_01_target_distribution")

    best = tbl.loc[tbl["rmse"].idxmin(), "target"]
    log.info("Chosen target: %s  (lower val RMSE: %.2f vs %.2f)",
             best,
             tbl.loc[tbl["target"] == best, "rmse"].values[0],
             tbl.loc[tbl["target"] != best, "rmse"].values[0])
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 5. DECISION TREE OVERFITTING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def dt_overfitting_analysis(X_train, y_train, X_val, y_val,
                            use_log: bool = False) -> None:
    """
    All RMSE values are reported in original price units (€) regardless of
    whether models are trained on log1p(y).  Back-transform is applied before
    computing errors so train vs val curves are comparable.
    """
    log.info("=" * 62)
    log.info("STEP 5 — Decision Tree overfitting analysis")
    log.info("=" * 62)

    # ground-truth in original price units
    y_train_orig = np.expm1(y_train) if use_log else y_train
    y_val_orig   = y_val                    # val is always passed in original units

    depths   = list(range(1, 21))
    tr_rmse  = []
    val_rmse = []

    for d in depths:
        pipe = Pipeline([
            ("prep", build_reg_preprocessor()),
            ("reg",  DecisionTreeRegressor(max_depth=d, random_state=RS)),
        ])
        pipe.fit(X_train, y_train)
        # back-transform predictions to original scale before computing RMSE
        tr_pred  = pipe.predict(X_train)
        val_pred = pipe.predict(X_val)
        if use_log:
            tr_pred  = np.expm1(tr_pred)
            val_pred = np.expm1(val_pred)
        tr_rmse.append(np.sqrt(mean_squared_error(y_train_orig, tr_pred)))
        val_rmse.append(np.sqrt(mean_squared_error(y_val_orig,  val_pred)))

    best_val_depth = depths[int(np.argmin(val_rmse))]
    log.info(
        "DT best val depth=%d (val RMSE=%.2f)  train RMSE=%.2f",
        best_val_depth, min(val_rmse), tr_rmse[best_val_depth - 1],
    )
    log.info(
        "DT at depth=20: train RMSE=%.2f  val RMSE=%.2f  (gap=%.2f)",
        tr_rmse[-1], val_rmse[-1], val_rmse[-1] - tr_rmse[-1],
    )

    ov_tbl = pd.DataFrame({
        "max_depth":  depths,
        "train_rmse": [round(v, 3) for v in tr_rmse],
        "val_rmse":   [round(v, 3) for v in val_rmse],
        "gap":        [round(v - t, 3) for t, v in zip(tr_rmse, val_rmse)],
    })
    save_table(ov_tbl, "rg_03_dt_overfitting")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, tr_rmse,  "o-", color=GREEN, lw=2, label="Train RMSE")
    ax.plot(depths, val_rmse, "s-", color=BRAND, lw=2, label="Val RMSE")
    ax.axvline(best_val_depth, color=GREY, ls="--", lw=1.5,
               label=f"Best val depth={best_val_depth}")
    ax.fill_between(
        depths, tr_rmse, val_rmse,
        where=[v > t for t, v in zip(tr_rmse, val_rmse)],
        alpha=0.12, color=BRAND, label="Overfit gap",
    )
    ax.set_title("Decision Tree Overfitting — Price Regression", fontweight="bold")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("RMSE (€)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "rg_02_dt_overfitting")

    return best_val_depth


# ══════════════════════════════════════════════════════════════════════════════
# 6. XGBoost HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train) -> Pipeline:
    log.info("=" * 62)
    log.info("STEP 6 — XGBoost hyperparameter tuning (30 iter, KFold-5)")
    log.info("=" * 62)

    pipe = Pipeline([
        ("prep", build_reg_preprocessor()),
        ("reg",  XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RS,
            verbosity=0,
            device="cpu",
        )),
    ])

    param_dist = {
        "reg__n_estimators":     [100, 200, 300, 400, 500],
        "reg__max_depth":        [3, 4, 5, 6, 7],
        "reg__learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
        "reg__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg__min_child_weight": [1, 3, 5, 7, 10],
        "reg__gamma":            [0, 0.05, 0.1, 0.2, 0.3],
        "reg__reg_alpha":        [0, 0.01, 0.1, 0.5, 1.0],
        "reg__reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RS),
        n_jobs=-1,
        random_state=RS,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params = {
        k.replace("reg__", ""): v
        for k, v in search.best_params_.items()
    }
    best_cv_rmse = -search.best_score_
    log.info("Best CV RMSE: %.4f", best_cv_rmse)
    log.info("Best params: %s", best_params)

    params_tbl = pd.DataFrame([
        {"parameter": k, "value": v} for k, v in best_params.items()
    ] + [{"parameter": "cv_rmse (5-fold)", "value": round(best_cv_rmse, 4)}])
    save_table(params_tbl, "rg_04_xgb_best_params")

    return search.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# 7. BUILD & TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def build_and_train(
    X_train, y_train, X_val, y_val,
    use_log: bool,
    best_dt_depth: int,
) -> dict:
    """
    Trains Dummy, Ridge, DT (best-depth), XGBoost (tuned) on the chosen target.
    Returns dict of {name: fitted_pipeline}.
    Evaluates each on val set in original price units.
    """
    log.info("=" * 62)
    log.info("STEP 4 — Train all models (target=%s)",
             "log1p(y)" if use_log else "raw y")
    log.info("=" * 62)

    y_tr = np.log1p(y_train) if use_log else y_train
    y_va = y_val                                    # always original scale for display

    def fit_eval(pipe, name):
        pipe.fit(X_train, y_tr)
        y_pred_va = pipe.predict(X_val)
        if use_log:
            y_pred_va = np.expm1(y_pred_va)
        m = reg_metrics(y_va, y_pred_va, label=f"[val] {name}")
        return pipe, m

    fitted = {}
    val_results = []

    # DummyRegressor
    dummy_pipe = Pipeline([
        ("prep", build_reg_preprocessor()),
        ("reg",  DummyRegressor(strategy="mean")),
    ])
    fitted["DummyRegressor"], m = fit_eval(dummy_pipe, "DummyRegressor")
    val_results.append({"model": "DummyRegressor", **m})

    # Ridge
    ridge_pipe = Pipeline([
        ("prep", build_reg_preprocessor()),
        ("reg",  Ridge(alpha=10.0)),
    ])
    fitted["Ridge"], m = fit_eval(ridge_pipe, "Ridge")
    val_results.append({"model": "Ridge", **m})

    # Decision Tree (best depth from overfitting analysis)
    dt_pipe = Pipeline([
        ("prep", build_reg_preprocessor()),
        ("reg",  DecisionTreeRegressor(max_depth=best_dt_depth, random_state=RS)),
    ])
    fitted[f"DecisionTree(d={best_dt_depth})"], m = fit_eval(
        dt_pipe, f"DecisionTree(d={best_dt_depth})"
    )
    val_results.append({"model": f"DecisionTree(d={best_dt_depth})", **m})

    # XGBoost tuned
    log.info("  Tuning XGBoost …")
    xgb_best = tune_xgboost(X_train, y_tr)
    y_pred_xgb_va = xgb_best.predict(X_val)
    if use_log:
        y_pred_xgb_va = np.expm1(y_pred_xgb_va)
    m_xgb = reg_metrics(y_va, y_pred_xgb_va, label="[val] XGBoost (tuned)")
    fitted["XGBoost (tuned)"] = xgb_best
    val_results.append({"model": "XGBoost (tuned)", **m_xgb})

    val_tbl = pd.DataFrame(val_results).sort_values("rmse")
    log.info("Validation leaderboard:\n%s", val_tbl.to_string(index=False))
    save_table(val_tbl, "rg_05_val_leaderboard")

    return fitted


# ══════════════════════════════════════════════════════════════════════════════
# 8. TEST-SET EVALUATION  (test set opened ONCE)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_test_set(
    fitted: dict,
    X_train, y_train,
    X_test,  y_test,
    use_log: bool,
) -> pd.DataFrame:
    log.info("=" * 62)
    log.info("STEP 7 — Test-set evaluation (test set opened here for the first time)")
    log.info("=" * 62)

    rows = []
    for name, pipe in fitted.items():
        # train metrics (in original units)
        y_pred_tr = pipe.predict(X_train)
        if use_log:
            y_pred_tr = np.expm1(y_pred_tr)
        m_tr = reg_metrics(y_train, y_pred_tr)

        # test metrics
        y_pred_te = pipe.predict(X_test)
        if use_log:
            y_pred_te = np.expm1(y_pred_te)
        m_te = reg_metrics(y_test, y_pred_te, label=f"[test] {name}")

        rows.append({
            "model":      name,
            "train_r2":   m_tr["r2"],
            "train_mae":  m_tr["mae"],
            "train_rmse": m_tr["rmse"],
            "test_r2":    m_te["r2"],
            "test_mae":   m_te["mae"],
            "test_rmse":  m_te["rmse"],
        })

    leaderboard = pd.DataFrame(rows).sort_values("test_rmse")
    log.info("Test-set leaderboard:\n%s", leaderboard.to_string(index=False))
    save_table(leaderboard, "rg_06_leaderboard_test")
    return leaderboard


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_predictions(fitted: dict, X_test, y_test, use_log: bool) -> None:
    log.info("Plotting predicted vs actual …")
    n_models = len(fitted)
    ncols = 2
    nrows = (n_models + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    axes = axes.flatten()

    colors = [GREY, BLUE, GREEN, BRAND]
    for ax, (name, pipe), color in zip(axes, fitted.items(), colors):
        y_pred = pipe.predict(X_test)
        if use_log:
            y_pred = np.expm1(y_pred)
        m = reg_metrics(y_test, y_pred)

        ax.scatter(y_test, y_pred, alpha=0.25, s=10, color=color)
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect")
        ax.set_title(
            f"{name}\nR²={m['r2']:.4f}  MAE={m['mae']:.1f}  RMSE={m['rmse']:.1f}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Actual price (€)")
        ax.set_ylabel("Predicted price (€)")
        ax.spines[["top", "right"]].set_visible(False)

    # hide spare subplot if odd number
    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("Predicted vs Actual — Room Price, Test Set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "rg_03_predicted_vs_actual")


def plot_residuals(fitted: dict, X_test, y_test, use_log: bool) -> None:
    log.info("Plotting residuals for XGBoost …")
    pipe = fitted.get("XGBoost (tuned)")
    if pipe is None:
        return

    y_pred = pipe.predict(X_test)
    if use_log:
        y_pred = np.expm1(y_pred)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.25, s=10, color=BRAND)
    axes[0].axhline(0, color="k", lw=1.2, ls="--")
    axes[0].set_title("Residuals vs Predicted", fontweight="bold")
    axes[0].set_xlabel("Predicted price (€)")
    axes[0].set_ylabel("Residual (€)")
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. Residual histogram
    axes[1].hist(residuals, bins=60, color=BRAND, alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="k", lw=1.2, ls="--")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual (€)")
    axes[1].set_ylabel("Count")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].text(
        0.97, 0.94,
        f"mean={residuals.mean():.1f}\nstd={residuals.std():.1f}",
        ha="right", va="top", transform=axes[1].transAxes, fontsize=9, color=GREY,
    )

    # 3. QQ plot
    from scipy.stats import probplot
    probplot(residuals, plot=axes[2])
    axes[2].get_lines()[0].set(color=BRAND, alpha=0.5, markersize=3)
    axes[2].get_lines()[1].set(color="k", lw=1.5)
    axes[2].set_title("QQ Plot of Residuals", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle("XGBoost (tuned) — Residual Analysis, Test Set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "rg_04_residuals_xgb")


def plot_feature_importances(fitted: dict, X_train) -> None:
    log.info("Plotting XGBoost feature importances …")
    pipe = fitted.get("XGBoost (tuned)")
    if pipe is None:
        return

    prep   = pipe.named_steps["prep"]
    xgb    = pipe.named_steps["reg"]
    feat_names = prep.named_steps["col"].get_feature_names_out()
    importances = xgb.feature_importances_

    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    save_table(imp_df, "rg_07_feature_importances_xgb")

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        imp_df["feature"][::-1],
        imp_df["importance"][::-1],
        color=BRAND, alpha=0.85, edgecolor="white",
    )
    ax.set_title("XGBoost Feature Importances (Gain) — Price Regression",
                 fontweight="bold")
    ax.set_xlabel("Importance (gain)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "rg_05_feature_importances")


def plot_leaderboard_bar(leaderboard: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [("test_rmse", "RMSE (€) — lower is better"),
               ("test_mae",  "MAE (€) — lower is better"),
               ("test_r2",   "R² — higher is better")]
    colors  = [BRAND, BLUE, GREEN]

    for ax, (col, title), color in zip(axes, metrics, colors):
        vals = leaderboard[col]
        bars = ax.bar(leaderboard["model"], vals, color=color, alpha=0.85,
                      width=0.5, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel(col.split("_", 1)[1].upper())
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Test-Set Leaderboard — Room Price Regression",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "rg_06_leaderboard_bar")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("RoomRadar — Regression Pipeline — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Data
    X, y = prepare_regression_data()

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_regression(X, y)

    # 3. Target comparison (val only)
    best_target = compare_targets(X_train, y_train, X_val, y_val)
    use_log = (best_target == "log1p")
    log.info("Using %s target for all models.", "log1p(y)" if use_log else "raw y")

    # 4. DT overfitting analysis
    best_dt_depth = dt_overfitting_analysis(
        X_train,
        np.log1p(y_train) if use_log else y_train,   # model trains on this
        X_val,
        y_val,                                         # always original price units
        use_log=use_log,
    )

    # 5. Train all models
    fitted = build_and_train(
        X_train, y_train, X_val, y_val,
        use_log=use_log,
        best_dt_depth=best_dt_depth,
    )

    # 6. Test-set evaluation
    leaderboard = evaluate_test_set(
        fitted, X_train, y_train, X_test, y_test, use_log=use_log
    )

    # 7. Plots
    plot_predictions(fitted, X_test, y_test, use_log)
    plot_residuals(fitted, X_test, y_test, use_log)
    plot_feature_importances(fitted, X_train)
    plot_leaderboard_bar(leaderboard)

    log.info("=" * 62)
    log.info("DONE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info(
        "Tables (%d new): %s",
        len(list(TABLE_DIR.glob("rg_*.csv"))), TABLE_DIR,
    )
    log.info(
        "Figures (%d new): %s",
        len(list(FIG_DIR.glob("rg_*.png"))), FIG_DIR,
    )
    log.info("=" * 62)


if __name__ == "__main__":
    main()
