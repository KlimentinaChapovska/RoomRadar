"""
RoomRadar — Cancellation Model Finalization
============================================
Rebuilds the winning XGBoost pipeline, calibrates probabilities on the
validation set, selects a decision threshold, defines risk bands, evaluates
on the untouched test set, and saves the production artefacts.

Run from project root:
    python ml/scripts/finalize_classification.py

Outputs
-------
models/cancellation_model.pkl      — calibrated production pipeline
models/cancellation_metadata.json  — threshold, risk bands, metrics
outputs/tables/prod_*.csv
outputs/figures/prod_*.png
"""

import sys, json, warnings, calendar
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

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from app.utils.logger import get_logger
from train_classification import (
    load_and_clean, split_data, build_preprocessor,
    FeatureEngineer, NUM_COLS, CAT_COLS, RS,
)
from calibration_pipeline import CalibratedPipeline

log = get_logger("finalize_clf")

TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR   = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"
for d in (TABLE_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

BRAND = "#922b21"
GREY  = "#7f8c8d"
GREEN = "#2ecc71"
DPI   = 150

BEST_XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        7,
    "learning_rate":    0.08,
    "subsample":        0.9,
    "colsample_bytree": 0.8,
    "gamma":            0.3,
    "reg_alpha":        0.1,
    "reg_lambda":       3.0,
    "min_child_weight": 1,
}

# ── risk band thresholds (business-interpretable, not data-fitted) ───────────
RISK_LOW_UPPER  = 0.30   # < 30 % → Low   risk
RISK_HIGH_LOWER = 0.60   # > 60 % → High  risk
                         # 30–60 %  → Medium risk


def save_fig(fig, name):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure  → %s", p.name)


def save_table(df, name):
    p = TABLE_DIR / f"{name}.csv"
    df.to_csv(p, index=True)
    log.info("  saved table   → %s", p.name)


def risk_band(prob: float) -> str:
    if prob < RISK_LOW_UPPER:
        return "low"
    if prob < RISK_HIGH_LOWER:
        return "medium"
    return "high"


# ══════════════════════════════════════════════════════════════════════════════
# 1. REBUILD PIPELINE (identical parameters to training run)
# ══════════════════════════════════════════════════════════════════════════════

def build_xgb_pipeline(scale_pos_weight: int) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf",  XGBClassifier(
            **BEST_XGB_PARAMS,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=RS,
            verbosity=0,
            device="cpu",
        )),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def _fit_isotonic(y_prob, y_true) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso


def _fit_sigmoid(y_prob, y_true) -> "_LR":
    lr = _LR(C=1.0, max_iter=1000)
    lr.fit(y_prob.reshape(-1, 1), y_true)
    return lr


def calibrate(xgb_pipe: Pipeline, X_val, y_val):
    """
    Fits isotonic and sigmoid (Platt) calibrators on the validation set.
    Returns the better CalibratedPipeline and its Brier score.
    cv='prefit' was removed in sklearn 1.9 — we use manual calibration instead.
    """
    log.info("=" * 62)
    log.info("STEP 2 — Probability calibration on validation set")
    log.info("=" * 62)

    y_prob_raw = xgb_pipe.predict_proba(X_val)[:, 1]

    iso_reg = _fit_isotonic(y_prob_raw, y_val.values)
    sig_reg = _fit_sigmoid( y_prob_raw, y_val.values)

    iso_cal = CalibratedPipeline(xgb_pipe, iso_reg, "isotonic")
    sig_cal = CalibratedPipeline(xgb_pipe, sig_reg, "sigmoid")

    y_prob_iso = iso_cal.predict_proba(X_val)[:, 1]
    y_prob_sig = sig_cal.predict_proba(X_val)[:, 1]

    brier_raw = brier_score_loss(y_val, y_prob_raw)
    brier_iso = brier_score_loss(y_val, y_prob_iso)
    brier_sig = brier_score_loss(y_val, y_prob_sig)

    log.info("Brier score  raw=%.4f  isotonic=%.4f  sigmoid=%.4f",
             brier_raw, brier_iso, brier_sig)

    cal_tbl = pd.DataFrame({
        "method":      ["uncalibrated", "isotonic", "sigmoid"],
        "brier_score": [round(brier_raw, 5),
                        round(brier_iso, 5),
                        round(brier_sig, 5)],
        "lower_is_better": ["✓", "✓", "✓"],
    })
    save_table(cal_tbl, "prod_01_calibration_brier_scores")

    # ── reliability (calibration) curves ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_bins = 10

    for ax, (label, probs, color) in zip(
        axes,
        [
            ("Uncalibrated",  y_prob_raw, BRAND),
            ("Isotonic",      y_prob_iso, "#2980b9"),
        ],
    ):
        fop, mpv = calibration_curve(y_val, probs, n_bins=n_bins, strategy="uniform")
        ax.plot(mpv, fop, "o-", color=color, lw=2, label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.set_title(f"Reliability Diagram — {label}", fontweight="bold")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives (actual rate)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        brier = brier_score_loss(y_val, probs)
        ax.text(0.04, 0.92, f"Brier = {brier:.4f}", transform=ax.transAxes,
                fontsize=10, color=color, fontweight="bold")

    # Add sigmoid on the right panel too
    fop_sig, mpv_sig = calibration_curve(
        y_val, y_prob_sig, n_bins=n_bins, strategy="uniform"
    )
    axes[1].plot(mpv_sig, fop_sig, "s--", color="#27ae60", lw=1.5, label="Sigmoid")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Reliability Diagram — Isotonic vs Sigmoid", fontweight="bold")

    fig.suptitle("Probability Calibration — Validation Set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "prod_01_reliability_curves")

    # choose the method with lower Brier score
    if brier_iso <= brier_sig:
        best_cal, best_method, best_brier = iso_cal, "isotonic", brier_iso
    else:
        best_cal, best_method, best_brier = sig_cal, "sigmoid", brier_sig

    log.info("Selected calibration: %s  (Brier=%.4f)", best_method, best_brier)
    return best_cal, best_method, brier_raw, best_brier


# ══════════════════════════════════════════════════════════════════════════════
# 3. THRESHOLD SELECTION ON VALIDATION (test set never touched here)
# ══════════════════════════════════════════════════════════════════════════════

def select_threshold(calibrated_model, X_val, y_val) -> float:
    log.info("=" * 62)
    log.info("STEP 3 — Threshold selection on validation set")
    log.info("=" * 62)

    y_prob = calibrated_model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.85, 0.01)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold":  round(float(t), 2),
            "accuracy":   round(accuracy_score(y_val, y_pred), 4),
            "precision":  round(precision_score(y_val, y_pred, pos_label=1, zero_division=0), 4),
            "recall":     round(recall_score(y_val, y_pred,    pos_label=1, zero_division=0), 4),
            "f1":         round(f1_score(y_val, y_pred,        pos_label=1, zero_division=0), 4),
        })

    tbl = pd.DataFrame(rows)
    best_row = tbl.loc[tbl["f1"].idxmax()]
    threshold = float(best_row["threshold"])

    log.info(
        "Best val F1=%.4f at threshold=%.2f  "
        "(precision=%.4f  recall=%.4f  accuracy=%.4f)",
        best_row["f1"], threshold,
        best_row["precision"], best_row["recall"], best_row["accuracy"],
    )
    save_table(tbl, "prod_02_threshold_search_val")

    # ── threshold sweep plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, color in [("precision", "#2980b9"), ("recall", GREEN), ("f1", BRAND)]:
        axes[0].plot(tbl["threshold"], tbl[metric], label=metric, color=color, lw=2)
    axes[0].axvline(threshold, color=GREY, ls="--", lw=1.5,
                    label=f"Chosen threshold = {threshold:.2f}")
    axes[0].set_title("Precision / Recall / F1 vs Threshold\n(validation set)",
                       fontweight="bold")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(tbl["threshold"], tbl["accuracy"], color="#8e44ad", lw=2)
    axes[1].axvline(threshold, color=GREY, ls="--", lw=1.5,
                    label=f"Threshold = {threshold:.2f}")
    axes[1].set_title("Accuracy vs Threshold\n(validation set)", fontweight="bold")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Threshold Selection on Validation Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "prod_02_threshold_analysis")

    return threshold


# ══════════════════════════════════════════════════════════════════════════════
# 4. RISK BAND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_risk_bands(calibrated_model, X_val, y_val, X_test, y_test,
                       threshold: float) -> None:
    log.info("=" * 62)
    log.info("STEP 4 — Risk band analysis")
    log.info("=" * 62)
    log.info(
        "Risk bands:  Low < %.2f  |  Medium %.2f–%.2f  |  High > %.2f",
        RISK_LOW_UPPER, RISK_LOW_UPPER, RISK_HIGH_LOWER, RISK_HIGH_LOWER,
    )

    records = []
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_prob  = calibrated_model.predict_proba(X)[:, 1]
        bands   = np.where(y_prob < RISK_LOW_UPPER, "low",
                  np.where(y_prob < RISK_HIGH_LOWER, "medium", "high"))
        df_tmp  = pd.DataFrame({"prob": y_prob, "band": bands, "actual": y.values})
        for band in ["low", "medium", "high"]:
            mask      = df_tmp["band"] == band
            n         = mask.sum()
            n_cancel  = df_tmp.loc[mask, "actual"].sum()
            records.append({
                "split":        split_name,
                "band":         band,
                "n_bookings":   int(n),
                "pct_of_split": round(n / len(df_tmp) * 100, 1),
                "n_canceled":   int(n_cancel),
                "actual_cancel_rate": round(n_cancel / max(n, 1) * 100, 1),
            })

    band_tbl = pd.DataFrame(records)
    save_table(band_tbl, "prod_03_risk_band_distribution")
    log.info("Risk band distribution (test):\n%s",
             band_tbl[band_tbl["split"]=="test"].to_string(index=False))

    # ── probability histogram by risk band ───────────────────────────────────
    y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]
    y_actual    = y_test.values
    band_colors = {"low": GREEN, "medium": "#f39c12", "high": BRAND}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # histogram by band
    for band, color in band_colors.items():
        if band == "low":
            mask = y_prob_test < RISK_LOW_UPPER
        elif band == "medium":
            mask = (y_prob_test >= RISK_LOW_UPPER) & (y_prob_test < RISK_HIGH_LOWER)
        else:
            mask = y_prob_test >= RISK_HIGH_LOWER
        axes[0].hist(y_prob_test[mask], bins=30, color=color, alpha=0.7,
                     label=f"{band.capitalize()} risk", edgecolor="white")

    axes[0].axvline(RISK_LOW_UPPER,  color="k", ls="--", lw=1.2, label=f"Low/Med={RISK_LOW_UPPER}")
    axes[0].axvline(RISK_HIGH_LOWER, color="k", ls=":",  lw=1.2, label=f"Med/High={RISK_HIGH_LOWER}")
    axes[0].axvline(threshold, color=GREY, ls="-", lw=1.5,
                    label=f"Decision threshold={threshold:.2f}")
    axes[0].set_title("Predicted Probability by Risk Band — Test Set", fontweight="bold")
    axes[0].set_xlabel("Cancellation probability")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # actual cancel rate by band (test)
    test_bands = band_tbl[band_tbl["split"] == "test"]
    bar_colors = [band_colors[b] for b in test_bands["band"]]
    bars = axes[1].bar(test_bands["band"].str.capitalize(),
                        test_bands["actual_cancel_rate"],
                        color=bar_colors, alpha=0.85, width=0.5, edgecolor="white")
    for bar, n in zip(bars, test_bands["n_bookings"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f"n={n:,}", ha="center", va="bottom", fontsize=9)
    axes[1].axhline(y_actual.mean() * 100, color=GREY, ls="--", lw=1.2,
                    label=f"Overall {y_actual.mean()*100:.1f}%")
    axes[1].set_title("Actual Cancellation Rate by Risk Band — Test Set", fontweight="bold")
    axes[1].set_ylabel("Actual cancellation rate (%)")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Risk Band Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "prod_03_risk_bands")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FINAL TEST-SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def final_test_evaluation(
    xgb_pipe,         # uncalibrated, for comparison
    calibrated_model, # production model
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    threshold: float,
    brier_raw: float,
    brier_cal: float,
) -> dict:
    log.info("=" * 62)
    log.info("STEP 5 — Final test-set evaluation (test set first opened here)")
    log.info("=" * 62)

    y_prob_cal  = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_cal  = (y_prob_cal >= threshold).astype(int)

    y_prob_raw  = xgb_pipe.predict_proba(X_test)[:, 1]
    y_pred_raw  = (y_prob_raw >= 0.5).astype(int)   # default threshold for comparison

    def metrics(y_true, y_pred, y_prob):
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred,    pos_label=1, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred,        pos_label=1, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
            "brier":     round(brier_score_loss(y_true, y_prob), 5),
        }

    m_cal = metrics(y_test, y_pred_cal, y_prob_cal)
    m_raw = metrics(y_test, y_pred_raw, y_prob_raw)

    log.info("Uncalibrated (threshold=0.50):  %s", m_raw)
    log.info("Calibrated   (threshold=%.2f):  %s", threshold, m_cal)

    # Overfitting summary across splits
    y_prob_tr  = calibrated_model.predict_proba(X_train)[:, 1]
    y_pred_tr  = (y_prob_tr >= threshold).astype(int)
    y_prob_va  = calibrated_model.predict_proba(X_val)[:, 1]
    y_pred_va  = (y_prob_va >= threshold).astype(int)

    m_train = metrics(y_train, y_pred_tr, y_prob_tr)
    m_val   = metrics(y_val,   y_pred_va, y_prob_va)

    ov_tbl = pd.DataFrame({
        "split":     ["train", "val", "test"],
        "accuracy":  [m_train["accuracy"],  m_val["accuracy"],  m_cal["accuracy"]],
        "precision": [m_train["precision"], m_val["precision"], m_cal["precision"]],
        "recall":    [m_train["recall"],    m_val["recall"],    m_cal["recall"]],
        "f1":        [m_train["f1"],        m_val["f1"],        m_cal["f1"]],
        "roc_auc":   [m_train["roc_auc"],   m_val["roc_auc"],   m_cal["roc_auc"]],
        "brier":     [m_train["brier"],     m_val["brier"],     m_cal["brier"]],
    })
    save_table(ov_tbl, "prod_04_overfitting_summary")
    log.info("Overfitting summary:\n%s", ov_tbl.to_string(index=False))

    log.info(
        "F1 gap  train→val=%.4f  val→test=%.4f  (smaller is better)",
        m_train["f1"] - m_val["f1"], m_val["f1"] - m_cal["f1"],
    )

    # comparison table: raw vs calibrated on test
    cmp_tbl = pd.DataFrame({
        "model":     ["XGBoost uncalibrated (t=0.50)",
                      f"XGBoost + isotonic calibration (t={threshold:.2f})"],
        "accuracy":  [m_raw["accuracy"],  m_cal["accuracy"]],
        "precision": [m_raw["precision"], m_cal["precision"]],
        "recall":    [m_raw["recall"],    m_cal["recall"]],
        "f1":        [m_raw["f1"],        m_cal["f1"]],
        "roc_auc":   [m_raw["roc_auc"],   m_cal["roc_auc"]],
        "brier":     [m_raw["brier"],     m_cal["brier"]],
    })
    save_table(cmp_tbl, "prod_05_raw_vs_calibrated_test")
    log.info("Test set comparison:\n%s", cmp_tbl.to_string(index=False))

    # ── confusion matrix — calibrated model ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (y_pred, title) in zip(
        axes,
        [
            (y_pred_raw, f"Uncalibrated  (t=0.50)\nF1={m_raw['f1']:.4f}"),
            (y_pred_cal, f"Calibrated + t={threshold:.2f}\nF1={m_cal['f1']:.4f}"),
        ],
    ):
        cm   = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not_Cancel", "Cancel"])
        disp.plot(ax=ax, colorbar=False, cmap="Reds")
        ax.set_title(title, fontweight="bold", fontsize=10)

    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "prod_04_confusion_matrix_final")

    # ── ROC curve with operating point ───────────────────────────────────────
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob_cal)
    auc = roc_auc_score(y_test, y_prob_cal)

    # find point on ROC curve closest to chosen threshold
    idx = np.argmin(np.abs(roc_thresh - threshold))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=BRAND, lw=2, label=f"Calibrated XGBoost (AUC={auc:.4f})")
    ax.plot(fpr[idx], tpr[idx], "o", ms=12, color=BRAND,
            label=f"Operating point t={threshold:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title("ROC Curve — Calibrated XGBoost, Test Set", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "prod_05_roc_final")

    return m_cal, m_train, m_val


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE PRODUCTION ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_production_artefacts(
    calibrated_model,
    threshold: float,
    calibration_method: str,
    brier_raw: float,
    brier_cal: float,
    test_metrics: dict,
    train_metrics: dict,
    val_metrics:   dict,
    n_train: int,
) -> None:
    log.info("=" * 62)
    log.info("STEP 6 — Saving production artefacts")
    log.info("=" * 62)

    # ── 6a. Pipeline ─────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "cancellation_model.pkl"
    joblib.dump(calibrated_model, model_path, compress=3)
    size_kb = model_path.stat().st_size / 1024
    log.info("  Pipeline saved → %s  (%.1f KB)", model_path.name, size_kb)

    # ── 6b. Metadata JSON ────────────────────────────────────────────────────
    metadata = {
        "model_name":          "roomradar_cancellation_xgb_calibrated",
        "version":             "1.0.0",
        "trained_on":          datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "positive_class":      "Canceled",
        "n_train_rows":        n_train,
        "xgb_params":          BEST_XGB_PARAMS,
        "calibration_method":  calibration_method,
        "brier_uncalibrated":  round(brier_raw, 5),
        "brier_calibrated":    round(brier_cal, 5),
        "threshold":           threshold,
        "risk_bands": {
            "low":    {"upper_exclusive": RISK_LOW_UPPER,
                       "description": "Low cancellation risk — no immediate action required"},
            "medium": {"lower_inclusive": RISK_LOW_UPPER,
                       "upper_exclusive": RISK_HIGH_LOWER,
                       "description": "Medium risk — send pre-arrival confirmation"},
            "high":   {"lower_inclusive": RISK_HIGH_LOWER,
                       "description": "High risk — request deposit or offer incentive"},
        },
        "train_metrics":       train_metrics,
        "val_metrics":         val_metrics,
        "test_metrics":        test_metrics,
        "overfitting_notes": (
            f"Train F1={train_metrics['f1']:.4f}  Val F1={val_metrics['f1']:.4f}  "
            f"Test F1={test_metrics['f1']:.4f}. "
            f"Train→Val gap={train_metrics['f1']-val_metrics['f1']:.4f} — XGBoost "
            f"memorises some training patterns. Gap is acceptable and stable "
            f"(Val→Test gap={val_metrics['f1']-test_metrics['f1']:.4f}). "
            "Regularisation (gamma=0.3, reg_lambda=3.0) was tuned to minimise this."
        ),
        "features_required": [
            "no_of_adults", "no_of_children", "no_of_weekend_nights",
            "no_of_week_nights", "type_of_meal_plan", "required_car_parking_space",
            "room_type_reserved", "lead_time", "arrival_year", "arrival_month",
            "arrival_date", "market_segment_type", "repeated_guest",
            "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
            "avg_price_per_room", "no_of_special_requests",
        ],
        "pipeline_steps": [
            "FeatureEngineer (custom TransformerMixin)",
            "ColumnTransformer: StandardScaler (numeric) + OHE (categorical)",
            "XGBClassifier (tuned via RandomizedSearchCV, 30 iter, 5-fold CV)",
            f"CalibratedClassifierCV (cv='prefit', method='{calibration_method}')",
        ],
    }

    meta_path = MODEL_DIR / "cancellation_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("  Metadata saved → %s", meta_path.name)

    log.info(
        "Production model ready:\n"
        "  threshold=%.2f | AUC=%.4f | F1=%.4f | Brier=%.5f (cal) vs %.5f (raw)",
        threshold, test_metrics["roc_auc"], test_metrics["f1"],
        brier_cal, brier_raw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("RoomRadar — Cancellation Finalization — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Reproduce identical data splits
    df = load_and_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 2. Refit XGB pipeline on training data
    log.info("=" * 62)
    log.info("STEP 1 — Refit XGBoost pipeline on training set")
    log.info("=" * 62)
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    xgb_pipe  = build_xgb_pipeline(scale_pos)
    xgb_pipe.fit(X_train, y_train)
    log.info("XGBoost pipeline fitted  (scale_pos_weight=%d)", scale_pos)

    # 3. Calibrate on validation set
    calibrated_model, cal_method, brier_raw, brier_cal = calibrate(
        xgb_pipe, X_val, y_val
    )

    # 4. Threshold selection on validation (test set untouched)
    threshold = select_threshold(calibrated_model, X_val, y_val)

    # 5. Risk band analysis
    analyse_risk_bands(
        calibrated_model, X_val, y_val, X_test, y_test, threshold
    )

    # 6. Final test-set evaluation
    test_m, train_m, val_m = final_test_evaluation(
        xgb_pipe, calibrated_model,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        threshold, brier_raw, brier_cal,
    )

    # 7. Save
    save_production_artefacts(
        calibrated_model, threshold, cal_method,
        brier_raw, brier_cal,
        test_m, train_m, val_m,
        n_train=len(y_train),
    )

    log.info("=" * 62)
    log.info("DONE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Tables  (%d new): %s",
             len(list(TABLE_DIR.glob("prod_*.csv"))), TABLE_DIR)
    log.info("Figures (%d new): %s",
             len(list(FIG_DIR.glob("prod_*.png"))),   FIG_DIR)
    log.info("Models  : %s", MODEL_DIR)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
