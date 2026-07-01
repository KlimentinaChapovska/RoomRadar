"""
RoomRadar — Cancellation Classification Pipeline
=================================================
Applies approved cleaning rules, engineers features, trains four classifiers,
tunes XGBoost with cross-validation, evaluates on a held-out test set, and
saves all tables and figures under outputs/.

Run from project root:
    python ml/scripts/train_classification.py

Does NOT save the production model — results must be reviewed first.
Does NOT train the regression model.
"""

import sys, warnings, calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ── paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.utils.logger import get_logger

RAW_CSV   = ROOT / "data" / "raw" / "Hotel Reservations.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR   = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "models"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger("train_clf")

BRAND  = "#922b21"
GREEN  = "#2ecc71"
GREY   = "#7f8c8d"
PALETTE = {"Not_Canceled": GREEN, "Canceled": BRAND}
DPI    = 150
RS     = 42        # random_state everywhere

# ── column definitions (post-engineering) ───────────────────────────────────
NUM_COLS = [
    # raw numeric (kept or superseded — tree models benefit from both)
    "no_of_adults", "no_of_children",
    "no_of_weekend_nights", "no_of_week_nights",
    "lead_time", "arrival_month",
    "required_car_parking_space", "repeated_guest",
    "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
    "avg_price_per_room", "no_of_special_requests",
    # engineered
    "total_nights", "total_guests", "has_children",
    "log_lead_time", "log_price", "price_is_zero",
    "is_weekend_stay", "arrival_season", "arrival_dow",
    "has_special_request", "prior_cancel_rate",
]
CAT_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]


# ── helpers ─────────────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure  → %s", path.name)

def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=True)
    log.info("  saved table   → %s", path.name)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN  (approved cleaning rules from EDA)
# ══════════════════════════════════════════════════════════════════════════════

def _is_invalid_date(row: pd.Series) -> bool:
    try:
        return int(row["arrival_date"]) > calendar.monthrange(
            int(row["arrival_year"]), int(row["arrival_month"])
        )[1]
    except Exception:
        return True


def load_and_clean() -> pd.DataFrame:
    log.info("=" * 62)
    log.info("STEP 1 — Load & clean")
    log.info("=" * 62)

    df = pd.read_csv(RAW_CSV)
    log.info("Raw rows: %d", len(df))

    # Rule 1 — drop calendar-invalid dates
    bad_dates = df.apply(_is_invalid_date, axis=1)
    df = df[~bad_dates].reset_index(drop=True)
    log.info("After dropping invalid dates (-37): %d rows", len(df))

    # Rule 3/4 — flag anomalies (do NOT drop)
    df["is_zero_adults"]  = (df["no_of_adults"] == 0).astype(int)
    df["is_zero_nights"]  = (
        (df["no_of_weekend_nights"] + df["no_of_week_nights"]) == 0
    ).astype(int)
    df["price_is_zero"]   = (df["avg_price_per_room"] == 0).astype(int)

    log.info(
        "Anomaly flags: zero_adults=%d  zero_nights=%d  price_zero=%d",
        df["is_zero_adults"].sum(), df["is_zero_nights"].sum(),
        df["price_is_zero"].sum(),
    )

    # Encode target: Canceled=1, Not_Canceled=0
    df["label"] = (df["booking_status"] == "Canceled").astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic feature derivation — safe to use inside a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["total_nights"]      = X["no_of_weekend_nights"] + X["no_of_week_nights"]
        X["total_guests"]      = X["no_of_adults"] + X["no_of_children"]
        X["has_children"]      = (X["no_of_children"] > 0).astype(int)
        X["log_lead_time"]     = np.log1p(X["lead_time"])
        X["log_price"]         = np.log1p(X["avg_price_per_room"])
        X["price_is_zero"]     = (X["avg_price_per_room"] == 0).astype(int)
        X["is_weekend_stay"]   = (X["no_of_weekend_nights"] > 0).astype(int)
        X["has_special_request"] = (X["no_of_special_requests"] > 0).astype(int)
        X["prior_cancel_rate"] = (
            X["no_of_previous_cancellations"]
            / (X["no_of_previous_cancellations"]
               + X["no_of_previous_bookings_not_canceled"] + 1e-9)
        )

        # Season: 0=Winter 1=Spring 2=Summer 3=Autumn
        season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
        X["arrival_season"] = X["arrival_month"].map(season_map).fillna(0).astype(int)

        # Day-of-week from calendar date
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

        # Drop identifiers and columns superseded by engineered versions
        drop = ["Booking_ID", "arrival_year", "arrival_date",
                "booking_status", "label",
                "is_zero_adults", "is_zero_nights"]
        return X.drop(columns=[c for c in drop if c in X.columns])


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor() -> Pipeline:
    """
    Two-stage: FeatureEngineer → ColumnTransformer.
    Numeric: StandardScaler (benefits LR; harmless for trees).
    Categorical: OHE with infrequent-category handling (min_frequency=10).
    """
    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
                CAT_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("eng", FeatureEngineer()), ("col", col_transformer)])


# ══════════════════════════════════════════════════════════════════════════════
# 4. STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    log.info("=" * 62)
    log.info("STEP 2 — Stratified train / val / test split  (60/20/20, RS=%d)", RS)
    log.info("=" * 62)

    feature_cols = [c for c in df.columns if c not in ("label", "booking_status")]
    X = df[feature_cols]
    y = df["label"]

    # 80 % train+val, 20 % test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RS
    )
    # 75 % of 80 % → 60 % train, 25 % of 80 % → 20 % val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RS
    )

    for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        n  = len(ys)
        nc = ys.sum()
        log.info(
            "  %-6s  %6d rows  |  Canceled %5d (%.1f%%)  Not_Canceled %5d (%.1f%%)",
            name, n, nc, nc/n*100, n-nc, (n-nc)/n*100,
        )

    split_tbl = pd.DataFrame({
        "split":        ["train", "val", "test"],
        "n_rows":       [len(y_train), len(y_val), len(y_test)],
        "n_canceled":   [y_train.sum(), y_val.sum(), y_test.sum()],
        "pct_canceled": [
            round(y_train.mean()*100,2),
            round(y_val.mean()*100,2),
            round(y_test.mean()*100,2),
        ],
    })
    save_table(split_tbl, "cl_01_split_sizes")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 5. DECISION TREE — OVERFITTING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def dt_overfitting_analysis(prep, X_train, y_train, X_val, y_val) -> int:
    log.info("=" * 62)
    log.info("STEP 3 — Decision Tree overfitting analysis")
    log.info("=" * 62)

    depths = list(range(1, 21))
    train_acc, val_acc   = [], []
    train_f1,  val_f1    = [], []

    for d in depths:
        pipe = Pipeline([
            ("prep", prep),
            ("clf",  DecisionTreeClassifier(max_depth=d, random_state=RS,
                                            class_weight="balanced")),
        ])
        pipe.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, pipe.predict(X_train)))
        val_acc.append(  accuracy_score(y_val,   pipe.predict(X_val)))
        train_f1.append( f1_score(y_train, pipe.predict(X_train), pos_label=1))
        val_f1.append(   f1_score(y_val,   pipe.predict(X_val),   pos_label=1))

    best_depth = depths[int(np.argmax(val_f1))]
    log.info("Best val F1 at max_depth=%d  (val_F1=%.4f)", best_depth, max(val_f1))

    ov_tbl = pd.DataFrame({
        "max_depth": depths,
        "train_accuracy": train_acc, "val_accuracy": val_acc,
        "train_f1": train_f1,        "val_f1":      val_f1,
    })
    save_table(ov_tbl, "cl_02_dt_overfitting")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tr, va, metric) in zip(
        axes,
        [(train_acc, val_acc, "Accuracy"), (train_f1, val_f1, "F1")],
    ):
        ax.plot(depths, tr, "o-", color=BRAND,  lw=2, label="Train")
        ax.plot(depths, va, "s-", color="#2980b9", lw=2, label="Validation")
        ax.axvline(best_depth, color=GREY, ls="--", lw=1.3,
                   label=f"Best depth={best_depth}")
        ax.set_title(f"Decision Tree — {metric} vs max_depth", fontweight="bold")
        ax.set_xlabel("max_depth")
        ax.set_ylabel(metric)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Decision Tree: Training vs Validation Performance\n"
                 "(gap shows overfitting beyond optimal depth)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "cl_01_dt_overfitting")

    return best_depth


# ══════════════════════════════════════════════════════════════════════════════
# 6. XGBOOST — HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

def tune_xgboost(prep, X_train, y_train) -> tuple[dict, Pipeline]:
    log.info("=" * 62)
    log.info("STEP 4 — XGBoost hyperparameter tuning  (RandomizedSearchCV, n_iter=30)")
    log.info("=" * 62)

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    log.info("scale_pos_weight = %d  (handles class imbalance)", scale_pos)

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos,
        random_state=RS,
        verbosity=0,
        device="cpu",
    )
    pipe = Pipeline([("prep", prep), ("clf", xgb_base)])

    param_dist = {
        "clf__n_estimators":     [100, 200, 300, 400, 500],
        "clf__max_depth":        [3, 4, 5, 6, 7],
        "clf__learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
        "clf__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight": [1, 3, 5, 7, 10],
        "clf__gamma":            [0, 0.05, 0.1, 0.2, 0.3],
        "clf__reg_alpha":        [0, 0.01, 0.1, 0.5, 1.0],
        "clf__reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        verbose=0,
        n_jobs=-1,
        random_state=RS,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    log.info("Best CV ROC-AUC: %.4f", search.best_score_)
    log.info("Best params: %s", best_params)

    # Save CV results summary
    cv_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        [["mean_test_score", "std_test_score", "rank_test_score",
          "param_clf__n_estimators", "param_clf__max_depth",
          "param_clf__learning_rate", "param_clf__subsample"]]
        .head(10)
        .round(4)
    )
    save_table(cv_df, "cl_03_xgb_cv_results")

    params_tbl = pd.DataFrame(best_params.items(), columns=["param", "value"])
    save_table(params_tbl, "cl_04_xgb_best_params")

    # Plot top-10 CV ROC-AUC
    top10 = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [f"Config {i+1}" for i in range(len(top10))][::-1],
        top10["mean_test_score"].values[::-1],
        xerr=top10["std_test_score"].values[::-1],
        color=BRAND, alpha=0.8, capsize=4,
    )
    ax.axvline(top10["mean_test_score"].iloc[0], color=GREY, ls="--", lw=1.2,
               label=f"Best {top10['mean_test_score'].iloc[0]:.4f}")
    ax.set_title("XGBoost RandomizedSearchCV — Top-10 CV ROC-AUC", fontweight="bold")
    ax.set_xlabel("Mean CV ROC-AUC (5-fold)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "cl_02_xgb_cv_results")

    return best_params, search.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# 7. BUILD ALL MODEL PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def build_pipelines(prep, best_depth: int, best_xgb_params: dict,
                    y_train: pd.Series) -> dict[str, Pipeline]:
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    # Quick XGB used only as the SelectFromModel feature selector
    quick_xgb = XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        random_state=RS, verbosity=0, device="cpu",
        eval_metric="auc",
    )

    return {
        "Dummy": Pipeline([
            ("prep", prep),
            ("clf",  DummyClassifier(strategy="most_frequent", random_state=RS)),
        ]),
        "Logistic Regression": Pipeline([
            ("prep", prep),
            ("sel",  SelectFromModel(quick_xgb, threshold="mean")),
            ("clf",  LogisticRegression(
                max_iter=1000, random_state=RS,
                class_weight="balanced", solver="lbfgs", C=1.0,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("prep", prep),
            ("clf",  DecisionTreeClassifier(
                max_depth=best_depth, random_state=RS, class_weight="balanced",
            )),
        ]),
        "XGBoost": Pipeline([
            ("prep", prep),
            ("clf",  XGBClassifier(
                **best_xgb_params,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=scale_pos,
                random_state=RS,
                verbosity=0,
                device="cpu",
            )),
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATE ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(pipeline: Pipeline, X, y, split_name: str) -> dict:
    y_pred  = pipeline.predict(X)
    y_prob  = (
        pipeline.predict_proba(X)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else np.zeros(len(y))
    )
    return {
        "split":     split_name,
        "accuracy":  round(accuracy_score(y, y_pred),               4),
        "precision": round(precision_score(y, y_pred, pos_label=1,
                                            zero_division=0),        4),
        "recall":    round(recall_score(y, y_pred, pos_label=1,
                                        zero_division=0),            4),
        "f1":        round(f1_score(y, y_pred, pos_label=1,
                                    zero_division=0),                4),
        "roc_auc":   round(roc_auc_score(y, y_prob)
                           if y_prob.any() else 0.5,                 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. SELECTED FEATURES (from LR pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def report_selected_features(lr_pipeline: Pipeline) -> None:
    try:
        all_names = (
            lr_pipeline.named_steps["prep"]
            .named_steps["col"]
            .get_feature_names_out()
        )
        mask     = lr_pipeline.named_steps["sel"].get_support()
        selected = all_names[mask]
        dropped  = all_names[~mask]
        log.info("Feature selection: %d / %d kept", mask.sum(), len(mask))
        log.info("  Selected:\n    %s", "\n    ".join(selected))

        tbl = pd.DataFrame({
            "feature": list(selected) + list(dropped),
            "selected": [True] * len(selected) + [False] * len(dropped),
        })
        save_table(tbl, "cl_05_selected_features")
    except Exception as exc:
        log.warning("Could not extract feature names: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 10. CONFUSION MATRICES PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(trained_pipes: dict, X_test, y_test) -> None:
    names = list(trained_pipes.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))

    for ax, name in zip(axes, names):
        pipe = trained_pipes[name]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not_Cancel", "Cancel"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Reds")
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\nAccuracy={acc:.3f}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "cl_03_confusion_matrices")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ROC CURVES PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(trained_pipes: dict, X_test, y_test) -> None:
    from sklearn.metrics import roc_curve
    colors = [BRAND, "#2980b9", "#27ae60", "#8e44ad"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for (name, pipe), color in zip(trained_pipes.items(), colors):
        if not hasattr(pipe, "predict_proba"):
            continue
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_title("ROC Curves — Test Set", fontweight="bold", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "cl_04_roc_curves")


# ══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(xgb_pipeline: Pipeline) -> None:
    try:
        feature_names = (
            xgb_pipeline.named_steps["prep"]
            .named_steps["col"]
            .get_feature_names_out()
        )
        importances = xgb_pipeline.named_steps["clf"].feature_importances_
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )
        save_table(fi, "cl_06_feature_importance")
        log.info("Top 10 XGBoost features:\n%s", fi.head(10).to_string(index=False))

        top20 = fi.head(20)
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(top20["feature"][::-1], top20["importance"][::-1],
                color=BRAND, alpha=0.8)
        ax.set_title("XGBoost Feature Importances (top 20)", fontweight="bold")
        ax.set_xlabel("Gain importance")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_fig(fig, "cl_05_feature_importance")
    except Exception as exc:
        log.warning("Feature importance plot failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 13. LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

def build_leaderboard(trained_pipes: dict,
                      X_train, y_train,
                      X_val,   y_val,
                      X_test,  y_test) -> pd.DataFrame:
    log.info("=" * 62)
    log.info("STEP 6 — Test-set evaluation & leaderboard")
    log.info("=" * 62)

    rows = []
    for name, pipe in trained_pipes.items():
        tr = evaluate(pipe, X_train, y_train, "train")
        va = evaluate(pipe, X_val,   y_val,   "val")
        te = evaluate(pipe, X_test,  y_test,  "test")
        rows.append({
            "model":          name,
            "train_accuracy": tr["accuracy"], "val_accuracy":  va["accuracy"],
            "test_accuracy":  te["accuracy"],
            "train_f1":       tr["f1"],        "val_f1":        va["f1"],
            "test_f1":        te["f1"],
            "train_roc_auc":  tr["roc_auc"],   "val_roc_auc":   va["roc_auc"],
            "test_roc_auc":   te["roc_auc"],
            "test_precision": te["precision"],
            "test_recall":    te["recall"],
        })
        log.info(
            "  %-22s | train F1 %.4f | val F1 %.4f | TEST F1 %.4f | "
            "TEST AUC %.4f",
            name, tr["f1"], va["f1"], te["f1"], te["roc_auc"],
        )

    board = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    save_table(board, "cl_07_leaderboard")
    return board


def plot_leaderboard(board: pd.DataFrame) -> None:
    metrics  = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    labels   = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors   = [BRAND, "#2980b9", "#27ae60", "#8e44ad", "#e67e22"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)
    for ax, col, label, color in zip(axes, metrics, labels, colors):
        vals = board[col].values
        names = board["model"].values
        bars = ax.barh(names[::-1], vals[::-1], color=color, alpha=0.8)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=8.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model Leaderboard — Test Set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "cl_06_leaderboard")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("RoomRadar Classification Pipeline — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ── 1. Data ──────────────────────────────────────────────────────────────
    df = load_and_clean()

    # ── 2. Split ─────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # ── 3. Preprocessor (shared; cloned per pipeline to avoid state leakage) ─
    from sklearn.base import clone
    def fresh_prep():
        return build_preprocessor()

    # ── 4. Decision Tree overfitting ─────────────────────────────────────────
    best_depth = dt_overfitting_analysis(
        fresh_prep(), X_train, y_train, X_val, y_val
    )

    # ── 5. XGBoost tuning ────────────────────────────────────────────────────
    best_xgb_params, best_xgb_pipeline = tune_xgboost(
        fresh_prep(), X_train, y_train
    )

    # ── 6. Build & train all final pipelines ─────────────────────────────────
    log.info("=" * 62)
    log.info("STEP 5 — Train all models on full training set")
    log.info("=" * 62)

    pipes = build_pipelines(fresh_prep(), best_depth, best_xgb_params, y_train)
    # Replace XGBoost with the already-tuned pipeline (also already fitted)
    pipes["XGBoost"] = best_xgb_pipeline

    trained = {}
    for name, pipe in pipes.items():
        if name == "XGBoost":
            trained[name] = pipe   # already fitted by RandomizedSearchCV
            log.info("  %-22s  already fitted (best from RandomizedSearchCV)", name)
            continue
        log.info("  fitting %-22s …", name)
        try:
            pipe.fit(X_train, y_train)
            trained[name] = pipe
        except Exception as exc:
            log.error("  FAILED: %s — %s", name, exc)

    # ── 7. Report selected features (from LR pipeline) ───────────────────────
    if "Logistic Regression" in trained:
        report_selected_features(trained["Logistic Regression"])

    # ── 8. Evaluate & leaderboard ─────────────────────────────────────────────
    board = build_leaderboard(
        trained, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrices(trained, X_test, y_test)
    plot_roc_curves(trained, X_test, y_test)
    plot_feature_importance(trained["XGBoost"])
    plot_leaderboard(board)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    log.info("=" * 62)
    log.info("DONE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Tables  → %s", TABLE_DIR)
    log.info("Figures → %s", FIG_DIR)
    log.info("=" * 62)
    log.info("\n%s", board[["model","test_accuracy","test_f1","test_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
