"""
RoomRadar — Exploratory Data Analysis
======================================
Loads data/raw/Hotel Reservations.csv, validates schema, runs full EDA,
saves tables to outputs/tables/ and charts to outputs/figures/.
Does NOT modify the raw CSV or train any model.

Run from project root:
    python ml/scripts/eda.py
"""

import sys
import warnings
import calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.utils.logger import get_logger

RAW_CSV   = ROOT / "data" / "raw" / "Hotel Reservations.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR   = ROOT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger("eda")

# ── style ──────────────────────────────────────────────────────────────────────
PALETTE = {"Not_Canceled": "#2ecc71", "Canceled": "#e74c3c"}
BRAND   = "#922b21"
GREY    = "#7f8c8d"
DPI     = 150

def save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure → %s", path.name)
    return path

def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path)
    log.info("  saved table  → %s", path.name)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_SCHEMA = {
    "Booking_ID":                            ("object", "str"),
    "no_of_adults":                          ("int64",),
    "no_of_children":                        ("int64",),
    "no_of_weekend_nights":                  ("int64",),
    "no_of_week_nights":                     ("int64",),
    "type_of_meal_plan":                     ("object", "str"),
    "required_car_parking_space":            ("int64",),
    "room_type_reserved":                    ("object", "str"),
    "lead_time":                             ("int64",),
    "arrival_year":                          ("int64",),
    "arrival_month":                         ("int64",),
    "arrival_date":                          ("int64",),
    "market_segment_type":                   ("object", "str"),
    "repeated_guest":                        ("int64",),
    "no_of_previous_cancellations":          ("int64",),
    "no_of_previous_bookings_not_canceled":  ("int64",),
    "avg_price_per_room":                    ("float64",),
    "no_of_special_requests":               ("int64",),
    "booking_status":                        ("object", "str"),
}

EXPECTED_CATS = {
    "type_of_meal_plan":     {"Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"},
    "room_type_reserved":    {f"Room_Type {i}" for i in range(1, 8)},
    "market_segment_type":   {"Online", "Offline", "Corporate", "Complementary", "Aviation"},
    "booking_status":        {"Not_Canceled", "Canceled"},
    "required_car_parking_space": {0, 1},
    "repeated_guest":        {0, 1},
}


def load_and_validate() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1 — Load & schema validation")
    log.info("=" * 60)

    df = pd.read_csv(RAW_CSV)
    log.info("Loaded %d rows × %d columns from %s", *df.shape, RAW_CSV.name)

    # column presence
    missing_cols = set(EXPECTED_SCHEMA) - set(df.columns)
    extra_cols   = set(df.columns) - set(EXPECTED_SCHEMA)
    if missing_cols:
        log.error("MISSING COLUMNS: %s", missing_cols)
    if extra_cols:
        log.warning("EXTRA COLUMNS (unexpected): %s", extra_cols)
    if not missing_cols and not extra_cols:
        log.info("All 19 expected columns present.")

    # dtypes — accept both "object" and "str" (pandas 2.x StringDtype alias)
    schema_errors = []
    for col, allowed in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual not in allowed:
            schema_errors.append(f"  {col}: expected one of {allowed}, got {actual}")
    if schema_errors:
        log.warning("Dtype mismatches:\n" + "\n".join(schema_errors))
    else:
        log.info("All column dtypes match expected schema.")

    # category membership
    cat_errors = []
    for col, allowed in EXPECTED_CATS.items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].unique())
        unexpected  = actual_vals - allowed
        if unexpected:
            cat_errors.append(f"  {col}: unexpected values {unexpected}")
    if cat_errors:
        log.warning("Unexpected categorical values:\n" + "\n".join(cat_errors))
    else:
        log.info("All categorical columns contain only expected values.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. MISSINGNESS, DUPLICATES, BASIC COUNTS
# ══════════════════════════════════════════════════════════════════════════════

def audit_completeness(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 2 — Missingness & duplicates")
    log.info("=" * 60)

    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        log.info("No missing values in any column.")
    else:
        log.warning("Missing values found:\n%s", missing[missing > 0].to_string())

    dupes = df.duplicated().sum()
    if dupes == 0:
        log.info("No duplicate rows.")
    else:
        log.warning("%d duplicate rows found.", dupes)

    summary = pd.DataFrame({
        "dtype":   df.dtypes.astype(str),
        "n_null":  df.isnull().sum(),
        "n_unique": df.nunique(),
        "pct_null": (df.isnull().mean() * 100).round(2),
    })
    save_table(summary, "01_column_summary")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TARGET DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def analyse_target(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 3 — Target distribution")
    log.info("=" * 60)

    counts = df["booking_status"].value_counts()
    pcts   = df["booking_status"].value_counts(normalize=True) * 100
    tbl = pd.DataFrame({"count": counts, "pct": pcts.round(2)})
    save_table(tbl, "02_target_distribution")
    log.info("booking_status distribution:\n%s", tbl.to_string())
    log.info(
        "Class imbalance ratio  Not_Canceled:Canceled = %.1f:1",
        counts["Not_Canceled"] / counts["Canceled"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = [PALETTE[s] for s in counts.index]

    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(axes[0].patches, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{v:,}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    axes[0].set_title("Booking Status Counts", fontweight="bold")
    axes[0].set_ylabel("Number of Bookings")
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Booking Status Share", fontweight="bold")

    fig.suptitle("Target Variable: Booking Status", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "01_target_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUSPICIOUS VALUE AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def audit_suspicious(df: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 4 — Suspicious value audit")
    log.info("=" * 60)

    flags: list[dict] = []

    def flag(col, mask, label, business_note):
        n = mask.sum()
        pct = n / len(df) * 100
        log.warning("  [%s] %s: %d rows (%.2f%%)", col, label, n, pct) if n > 0 \
            else log.info("  [%s] %s: 0 rows — OK", col, label)
        flags.append({"column": col, "issue": label, "n_rows": n,
                      "pct": round(pct, 3), "business_note": business_note})

    # --- adults ---
    flag("no_of_adults", df["no_of_adults"] == 0,
         "zero adults",
         "A hotel room with 0 adults is operationally invalid. "
         "Likely data-entry error. Consider filtering these rows before training.")
    flag("no_of_adults", df["no_of_adults"] > 4,
         "more than 4 adults",
         "Dataset max is 4; no rows found. Upper cap appears enforced.")

    # --- children ---
    flag("no_of_children", df["no_of_children"] > 5,
         "more than 5 children",
         "Extreme values (max 10). Possible data-entry error. "
         "Verify against room capacity; may need capping.")

    # --- stay length ---
    zero_nights = (df["no_of_weekend_nights"] + df["no_of_week_nights"]) == 0
    flag("no_of_weekend_nights + no_of_week_nights", zero_nights,
         "zero total nights",
         "Same-day bookings are unusual for a hotel. "
         "Could be day-use rooms or entry errors. Investigate before training.")

    # --- price ---
    flag("avg_price_per_room", df["avg_price_per_room"] == 0,
         "zero price",
         "Rooms priced at £0 are likely complimentary stays, staff bookings, "
         "or system errors. These should be excluded from the price regression model. "
         "For the cancellation model, keep but encode as a feature flag.")
    flag("avg_price_per_room", df["avg_price_per_room"] > 500,
         "price > 500",
         "Only 1 row; likely a genuine luxury booking or data error. Inspect individually.")

    q3 = df["avg_price_per_room"].quantile(0.75)
    iqr = q3 - df["avg_price_per_room"].quantile(0.25)
    upper_fence = q3 + 1.5 * iqr
    flag("avg_price_per_room", df["avg_price_per_room"] > upper_fence,
         f"IQR outlier (> {upper_fence:.1f})",
         "1,069 rows above IQR upper fence. These are real high-value bookings, "
         "not errors. Do NOT remove. Apply log-transform for the price model.")

    # --- lead time ---
    flag("lead_time", df["lead_time"] > 365,
         "lead time > 365 days (~1 year)",
         "102 rows booked more than a year ahead. Genuine for event/wedding bookings. "
         "Keep, but consider a log or sqrt transform.")

    # --- date validity ---
    def is_invalid_date(row):
        try:
            max_day = calendar.monthrange(int(row["arrival_year"]), int(row["arrival_month"]))[1]
            return int(row["arrival_date"]) > max_day
        except Exception:
            return True

    invalid_dates = df.apply(is_invalid_date, axis=1)
    flag("arrival_date", invalid_dates,
         "calendar-invalid date (e.g. Feb 30)",
         "Rows with dates that don't exist on a calendar. "
         "Likely data-entry error; drop or correct before training.")

    # --- year range ---
    log.info("  [arrival_year] unique values: %s", sorted(df["arrival_year"].unique()))
    flags.append({"column": "arrival_year", "issue": "only 2017–2018",
                  "n_rows": len(df), "pct": 100.0,
                  "business_note": "Narrow 2-year window. Model may not generalise to other years; "
                                   "use month + season features rather than raw year."})

    # --- binary columns ---
    for col in ("required_car_parking_space", "repeated_guest"):
        unexpected = ~df[col].isin([0, 1])
        flag(col, unexpected, "value not in {0,1}",
             "Should be binary. Unexpected values would be data corruption.")

    audit_df = pd.DataFrame(flags)
    save_table(audit_df, "03_suspicious_value_audit")
    return audit_df


# ══════════════════════════════════════════════════════════════════════════════
# 5. NUMERIC DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_numerics(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 5 — Numeric distributions")
    log.info("=" * 60)

    num_cols = [
        "lead_time", "avg_price_per_room", "no_of_adults",
        "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
        "no_of_special_requests", "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
    ]

    desc = df[num_cols].describe().T.round(2)
    desc["skewness"] = df[num_cols].skew().round(3)
    desc["kurtosis"] = df[num_cols].kurt().round(3)
    save_table(desc, "04_numeric_summary")
    log.info("Numeric summary (skew > |1| = notable):\n%s",
             desc[["mean", "std", "min", "50%", "max", "skewness"]].to_string())

    # --- price histogram + KDE + log ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    price = df["avg_price_per_room"]
    axes[0].hist(price, bins=80, color=BRAND, alpha=0.75, edgecolor="white")
    axes[0].axvline(price.median(), color="black", ls="--", lw=1.4, label=f"Median {price.median():.0f}")
    axes[0].axvline(price.mean(),   color=GREY,  ls="--", lw=1.4, label=f"Mean {price.mean():.0f}")
    axes[0].set_title("Room Price Distribution (raw)", fontweight="bold")
    axes[0].set_xlabel("avg_price_per_room (€)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    price_nz = price[price > 0]
    log_price = np.log1p(price_nz)
    axes[1].hist(log_price, bins=60, color="#2980b9", alpha=0.75, edgecolor="white")
    axes[1].axvline(log_price.median(), color="black", ls="--", lw=1.4)
    axes[1].set_title("Room Price Distribution (log1p, zero-price excluded)", fontweight="bold")
    axes[1].set_xlabel("log1p(avg_price_per_room)")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Price Distribution — Raw vs Log-Transformed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_price_distribution")

    # --- lead time -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    lt = df["lead_time"]
    axes[0].hist(lt, bins=70, color=BRAND, alpha=0.75, edgecolor="white")
    axes[0].axvline(lt.median(), color="black", ls="--", lw=1.4, label=f"Median {lt.median():.0f}d")
    axes[0].set_title("Lead Time Distribution (raw)", fontweight="bold")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    log_lt = np.log1p(lt)
    axes[1].hist(log_lt, bins=60, color="#8e44ad", alpha=0.75, edgecolor="white")
    axes[1].set_title("Lead Time Distribution (log1p)", fontweight="bold")
    axes[1].set_xlabel("log1p(lead_time)")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Lead Time — Raw vs Log-Transformed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_lead_time_distribution")

    # --- guests + stay length ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    guest_cols = [
        ("no_of_adults",          "Adults per Booking",   "#2980b9"),
        ("no_of_children",        "Children per Booking", "#e67e22"),
        ("no_of_weekend_nights",  "Weekend Nights",       "#27ae60"),
    ]
    for ax, (col, title, color) in zip(axes, guest_cols):
        vc = df[col].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Guest & Stay Composition", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_guests_and_stay")

    # --- special requests ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = df["no_of_special_requests"].value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, color=BRAND, alpha=0.8, edgecolor="white")
    ax.set_title("Special Requests per Booking", fontweight="bold")
    ax.set_xlabel("no_of_special_requests")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "05_special_requests")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CATEGORICAL PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def analyse_categories(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 6 — Categorical profiles")
    log.info("=" * 60)

    cat_cols = [
        "type_of_meal_plan", "room_type_reserved",
        "market_segment_type", "required_car_parking_space", "repeated_guest",
    ]
    records = []
    for col in cat_cols:
        vc = df[col].value_counts()
        pct = df[col].value_counts(normalize=True) * 100
        for val in vc.index:
            records.append({"column": col, "value": str(val),
                            "count": vc[val], "pct": round(pct[val], 2)})
    cat_tbl = pd.DataFrame(records)
    save_table(cat_tbl, "05_categorical_counts")
    log.info("Categorical counts saved.")

    # --- market segment + room type side by side ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mkt = df["market_segment_type"].value_counts()
    axes[0].barh(mkt.index[::-1], mkt.values[::-1], color=BRAND, alpha=0.8)
    axes[0].set_title("Market Segment", fontweight="bold")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(mkt.values[::-1]):
        axes[0].text(v + 150, i, f"{v:,}", va="center", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    rm = df["room_type_reserved"].value_counts()
    axes[1].barh(rm.index[::-1], rm.values[::-1], color="#2980b9", alpha=0.8)
    axes[1].set_title("Room Type Reserved", fontweight="bold")
    axes[1].set_xlabel("Count")
    for i, v in enumerate(rm.values[::-1]):
        axes[1].text(v + 100, i, f"{v:,}", va="center", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Booking Channel & Room Type", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "06_market_and_room_type")

    # --- meal plan ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    mp = df["type_of_meal_plan"].value_counts()
    ax.bar(mp.index, mp.values, color="#27ae60", alpha=0.8, edgecolor="white")
    for bar, v in zip(ax.patches, mp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Meal Plan Selection", fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "07_meal_plan")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CANCELLATION DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════

def analyse_cancellations(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 7 — Cancellation deep-dive")
    log.info("=" * 60)

    cancel_col = (df["booking_status"] == "Canceled").astype(int)

    def cancel_rate(group_col: str) -> pd.DataFrame:
        tbl = df.groupby(group_col)["booking_status"].apply(
            lambda x: (x == "Canceled").mean() * 100
        ).reset_index(name="cancel_rate_pct")
        tbl["n"] = df.groupby(group_col).size().values
        return tbl.sort_values("cancel_rate_pct", ascending=False)

    # --- by market segment ---------------------------------------------------
    mkt_cr = cancel_rate("market_segment_type")
    save_table(mkt_cr, "06_cancel_rate_by_market_segment")
    log.info("Cancel rate by market segment:\n%s", mkt_cr.to_string(index=False))

    # --- by room type --------------------------------------------------------
    rm_cr = cancel_rate("room_type_reserved")
    save_table(rm_cr, "07_cancel_rate_by_room_type")

    # --- by meal plan --------------------------------------------------------
    mp_cr = cancel_rate("type_of_meal_plan")
    save_table(mp_cr, "08_cancel_rate_by_meal_plan")

    # --- by special requests -------------------------------------------------
    sr_cr = cancel_rate("no_of_special_requests")
    save_table(sr_cr, "09_cancel_rate_by_special_requests")
    log.info("Cancel rate by special requests:\n%s", sr_cr.to_string(index=False))

    # --- chart: cancel rate by market + room type ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(mkt_cr["market_segment_type"][::-1],
                 mkt_cr["cancel_rate_pct"][::-1], color=BRAND, alpha=0.8)
    axes[0].axvline(cancel_col.mean() * 100, color=GREY, ls="--", lw=1.2,
                    label=f"Overall {cancel_col.mean()*100:.1f}%")
    axes[0].set_title("Cancellation Rate by Market Segment", fontweight="bold")
    axes[0].set_xlabel("Cancellation Rate (%)")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh(rm_cr["room_type_reserved"][::-1],
                 rm_cr["cancel_rate_pct"][::-1], color="#2980b9", alpha=0.8)
    axes[1].axvline(cancel_col.mean() * 100, color=GREY, ls="--", lw=1.2,
                    label=f"Overall {cancel_col.mean()*100:.1f}%")
    axes[1].set_title("Cancellation Rate by Room Type", fontweight="bold")
    axes[1].set_xlabel("Cancellation Rate (%)")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Where Cancellations Concentrate", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_cancel_rate_by_category")

    # --- chart: special requests vs cancel rate ------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sr_cr["no_of_special_requests"].astype(str),
           sr_cr["cancel_rate_pct"], color=BRAND, alpha=0.8, edgecolor="white")
    ax.axhline(cancel_col.mean() * 100, color=GREY, ls="--", lw=1.2,
               label=f"Overall {cancel_col.mean()*100:.1f}%")
    ax.set_title("Cancellation Rate by Number of Special Requests", fontweight="bold")
    ax.set_xlabel("no_of_special_requests")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "09_cancel_rate_by_special_requests")

    # --- chart: lead time vs cancellation ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for status, color in PALETTE.items():
        subset = df[df["booking_status"] == status]["lead_time"]
        axes[0].hist(subset, bins=60, alpha=0.6, color=color, label=status, density=True)
    axes[0].set_title("Lead Time by Booking Status", fontweight="bold")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # binned lead-time cancel rate
    df2 = df.copy()
    df2["lt_bin"] = pd.cut(df2["lead_time"], bins=[0, 14, 30, 60, 90, 180, 365, 500],
                           labels=["0–14", "15–30", "31–60", "61–90", "91–180", "181–365", "365+"])
    lt_cr = df2.groupby("lt_bin", observed=True)["booking_status"].apply(
        lambda x: (x == "Canceled").mean() * 100
    ).reset_index(name="cancel_rate_pct")
    axes[1].bar(lt_cr["lt_bin"].astype(str), lt_cr["cancel_rate_pct"],
                color=BRAND, alpha=0.8, edgecolor="white")
    axes[1].axhline(cancel_col.mean() * 100, color=GREY, ls="--", lw=1.2,
                    label=f"Overall {cancel_col.mean()*100:.1f}%")
    axes[1].set_title("Cancellation Rate by Lead-Time Bucket", fontweight="bold")
    axes[1].set_xlabel("Lead Time (days)")
    axes[1].set_ylabel("Cancellation Rate (%)")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    save_table(lt_cr, "10_cancel_rate_by_lead_time_bucket")
    fig.suptitle("Lead Time and Cancellation Risk", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "10_lead_time_vs_cancellation")

    # --- price by booking status (boxplot) -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    price_data = [
        df.loc[df["booking_status"] == s, "avg_price_per_room"].values
        for s in ["Not_Canceled", "Canceled"]
    ]
    bp = axes[0].boxplot(price_data, tick_labels=["Not_Canceled", "Canceled"],
                         patch_artist=True, widths=0.5,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Room Price by Booking Status", fontweight="bold")
    axes[0].set_ylabel("avg_price_per_room (€)")
    axes[0].spines[["top", "right"]].set_visible(False)

    # violin
    df_plot = df[["avg_price_per_room", "booking_status"]].copy()
    df_plot = df_plot[df_plot["avg_price_per_room"] > 0]  # exclude zero-price for violin
    sns.violinplot(
        data=df_plot, x="booking_status", y="avg_price_per_room",
        palette=PALETTE, inner="quartile", ax=axes[1]
    )
    axes[1].set_title("Price Density by Status (zero-price excluded)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("avg_price_per_room (€)")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Room Price vs Booking Outcome", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "11_price_vs_cancellation")

    # stat test
    g1 = df.loc[df["booking_status"] == "Not_Canceled", "avg_price_per_room"]
    g2 = df.loc[df["booking_status"] == "Canceled",     "avg_price_per_room"]
    stat, pval = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    log.info(
        "Mann-Whitney U (price Not_Canceled vs Canceled): U=%.0f, p=%.4e — %s",
        stat, pval,
        "SIGNIFICANT" if pval < 0.05 else "not significant"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════

def analyse_seasonality(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 8 — Seasonality")
    log.info("=" * 60)

    month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

    # --- bookings per month --------------------------------------------------
    monthly = df.groupby(["arrival_year", "arrival_month"])["booking_status"].agg(
        total="count",
        canceled=lambda x: (x == "Canceled").sum(),
    ).reset_index()
    monthly["cancel_rate"] = monthly["canceled"] / monthly["total"] * 100
    save_table(monthly, "11_monthly_volume_and_cancel_rate")
    log.info("Monthly volume and cancel rate saved.")

    # volume by month (stacked bar)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    month_status = (
        df.groupby(["arrival_month", "booking_status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Not_Canceled", "Canceled"])
    )
    month_status.index = [month_labels[m - 1] for m in month_status.index]
    month_status.plot(
        kind="bar", stacked=True, ax=axes[0],
        color=[PALETTE["Not_Canceled"], PALETTE["Canceled"]],
        edgecolor="white", width=0.7
    )
    axes[0].set_title("Monthly Booking Volume by Status", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Bookings")
    axes[0].legend(title="Status")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].spines[["top", "right"]].set_visible(False)

    # cancel rate by month
    cr_by_month = (
        df.groupby("arrival_month")["booking_status"]
        .apply(lambda x: (x == "Canceled").mean() * 100)
        .reset_index(name="cancel_rate_pct")
    )
    cr_by_month["month_label"] = [month_labels[m - 1] for m in cr_by_month["arrival_month"]]
    axes[1].plot(cr_by_month["month_label"], cr_by_month["cancel_rate_pct"],
                 marker="o", color=BRAND, linewidth=2)
    axes[1].fill_between(cr_by_month["month_label"], cr_by_month["cancel_rate_pct"],
                         alpha=0.15, color=BRAND)
    axes[1].axhline(df["booking_status"].eq("Canceled").mean() * 100,
                    color=GREY, ls="--", lw=1.2, label="Overall avg")
    axes[1].set_title("Monthly Cancellation Rate", fontweight="bold")
    axes[1].set_xlabel("Arrival Month")
    axes[1].set_ylabel("Cancellation Rate (%)")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Seasonal Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "12_seasonality")

    # --- avg price by month --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    price_month = df[df["avg_price_per_room"] > 0].groupby("arrival_month")["avg_price_per_room"]
    pm_mean = price_month.mean()
    pm_std  = price_month.std()
    x = range(1, 13)
    ax.plot([month_labels[m - 1] for m in x], pm_mean.values,
            marker="o", color=BRAND, linewidth=2)
    ax.fill_between(
        [month_labels[m - 1] for m in x],
        pm_mean.values - pm_std.values,
        pm_mean.values + pm_std.values,
        alpha=0.15, color=BRAND, label="±1 SD"
    )
    ax.set_title("Average Room Price by Arrival Month (zero-price excluded)", fontweight="bold")
    ax.set_xlabel("Arrival Month")
    ax.set_ylabel("avg_price_per_room (€)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "13_price_by_month")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CORRELATION & FEATURE RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_correlations(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 9 — Correlations")
    log.info("=" * 60)

    df2 = df.copy()
    df2["canceled"]   = (df2["booking_status"] == "Canceled").astype(int)
    df2["total_nights"] = df2["no_of_weekend_nights"] + df2["no_of_week_nights"]
    df2["total_guests"]  = df2["no_of_adults"] + df2["no_of_children"]

    num_cols = [
        "canceled", "lead_time", "avg_price_per_room",
        "no_of_adults", "no_of_children", "total_nights", "total_guests",
        "no_of_special_requests", "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled", "required_car_parking_space",
        "repeated_guest",
    ]
    corr = df2[num_cols].corr(method="spearman")
    save_table(corr, "12_spearman_correlation_matrix")

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman Correlation Matrix\n(lower triangle)", fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, "14_correlation_heatmap")

    # top correlations with target
    target_corr = corr["canceled"].drop("canceled").abs().sort_values(ascending=False)
    log.info("Top features correlated with cancellation (Spearman |r|):\n%s",
             target_corr.head(8).to_string())
    save_table(target_corr.reset_index().rename(columns={"index": "feature", "canceled": "|r|"}),
               "13_target_correlations")


# ══════════════════════════════════════════════════════════════════════════════
# 10. REPEATED GUESTS & PRIOR HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def analyse_guest_history(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("STEP 10 — Repeated guests & prior history")
    log.info("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    rg = df.groupby("repeated_guest")["booking_status"].apply(
        lambda x: (x == "Canceled").mean() * 100
    ).reset_index(name="cancel_rate_pct")
    rg["label"] = rg["repeated_guest"].map({0: "New Guest", 1: "Repeat Guest"})
    log.info("Cancel rate by repeated_guest:\n%s", rg.to_string(index=False))
    save_table(rg, "14_cancel_rate_by_repeated_guest")

    axes[0].bar(rg["label"], rg["cancel_rate_pct"],
                color=[PALETTE["Not_Canceled"], PALETTE["Canceled"]], alpha=0.8, width=0.4)
    for bar, v in zip(axes[0].patches, rg["cancel_rate_pct"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=11)
    axes[0].set_title("Cancellation Rate: New vs Repeat Guests", fontweight="bold")
    axes[0].set_ylabel("Cancellation Rate (%)")
    axes[0].spines[["top", "right"]].set_visible(False)

    # prior cancellations capped at 5 for readability
    df2 = df.copy()
    df2["prev_cancel_capped"] = df2["no_of_previous_cancellations"].clip(upper=5)
    pc_cr = df2.groupby("prev_cancel_capped")["booking_status"].apply(
        lambda x: (x == "Canceled").mean() * 100
    ).reset_index(name="cancel_rate_pct")
    pc_cr["label"] = pc_cr["prev_cancel_capped"].astype(str)
    pc_cr.loc[pc_cr["prev_cancel_capped"] == 5, "label"] = "5+"
    axes[1].bar(pc_cr["label"], pc_cr["cancel_rate_pct"],
                color=BRAND, alpha=0.8, edgecolor="white")
    axes[1].set_title("Cancel Rate by Prior Cancellations (capped at 5)", fontweight="bold")
    axes[1].set_xlabel("no_of_previous_cancellations")
    axes[1].set_ylabel("Cancellation Rate (%)")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Guest History Signals", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "15_guest_history")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ENGINEERED FEATURE DEFINITIONS (no training)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_DEFINITIONS = [
    # ── derived from raw columns ──────────────────────────────────────────────
    ("total_nights",        "no_of_weekend_nights + no_of_week_nights",
     "Total length of stay. Avoids the collinearity between the two raw columns."),
    ("total_guests",        "no_of_adults + no_of_children",
     "Party size. Collapses two correlated columns."),
    ("has_children",        "no_of_children > 0 → 0/1",
     "Binary flag; families may have distinct booking behaviour."),
    ("log_lead_time",       "log1p(lead_time)",
     "Heavily right-skewed raw value; log makes it near-normal for linear models."),
    ("log_price",           "log1p(avg_price_per_room)",
     "Price is right-skewed with zero-price anomalies; log compresses extremes."),
    ("price_is_zero",       "avg_price_per_room == 0 → 0/1",
     "Explicit flag for complimentary/error rows — preserves signal without distorting log transform."),
    ("is_weekend_stay",     "no_of_weekend_nights > 0 → 0/1",
     "Weekend vs pure weekday trips may differ in cancellation risk."),
    ("arrival_season",      "arrival_month → Spring/Summer/Autumn/Winter",
     "Collapses 12 months into 4 seasonal categories for sparsity reduction."),
    ("arrival_dow",         "Derived from arrival_year + arrival_month + arrival_date → day-of-week",
     "Day of week may signal leisure vs business travel."),
    ("has_parking",         "required_car_parking_space == 1 → 0/1",
     "Alias for readability; same encoding."),
    ("has_special_request", "no_of_special_requests > 0 → 0/1",
     "Binary: any request vs none. Guests with requests may be more committed."),
    ("prior_cancel_rate",   "no_of_previous_cancellations / (no_of_previous_cancellations + no_of_previous_bookings_not_canceled + 1e-9)",
     "Guest-level historical cancel rate. Smoothed denominator avoids div/0."),
    ("is_repeat_guest",     "repeated_guest (already 0/1)",
     "Kept as-is; renamed for clarity in feature set."),
    # ── target encodings (compute on train fold only) ─────────────────────────
    ("meal_plan_encoded",   "target_encode(type_of_meal_plan) on train fold",
     "Meal Plan 3 has only 5 rows — rare-category risk; target encoding handles it."),
    ("room_type_encoded",   "target_encode(room_type_reserved) on train fold",
     "Room_Type 3 has only 7 rows; ordinal or target encoding preferred."),
    ("market_seg_encoded",  "target_encode(market_segment_type) on train fold",
     "Aviation (125 rows) is sparse; target encoding is safer than one-hot."),
]


def save_feature_definitions() -> None:
    log.info("=" * 60)
    log.info("STEP 11 — Feature engineering definitions")
    log.info("=" * 60)
    tbl = pd.DataFrame(FEATURE_DEFINITIONS,
                       columns=["feature_name", "derivation", "rationale"])
    save_table(tbl, "15_feature_definitions")
    log.info("Defined %d engineered features (not applied — training step only).",
             len(FEATURE_DEFINITIONS))
    for name, deriv, rat in FEATURE_DEFINITIONS:
        log.info("  %-28s  ←  %s", name, deriv)


# ══════════════════════════════════════════════════════════════════════════════
# 12. CLEANING DECISION LOG
# ══════════════════════════════════════════════════════════════════════════════

CLEANING_DECISIONS = [
    ("Drop Booking_ID",
     "Sequential identifier; encodes no predictive signal and would leak row ordering into tree splits.",
     "APPLY — always drop before training."),
    ("Keep zero-adults rows (n=139)",
     "Operationally invalid, but we do NOT automatically remove them. Flag with is_zero_adults=1 "
     "and let the model learn their pattern. Revisit if model audit shows noise.",
     "FLAG, do not drop."),
    ("Keep zero-price rows (n=545) for cancellation model",
     "May represent complimentary stays. Add price_is_zero flag so the model can distinguish them "
     "from paid bookings. Log-transform will handle the rest.",
     "FLAG with price_is_zero; keep rows."),
    ("Exclude zero-price rows from price regression model",
     "Predicting a $0 price has no operational meaning. These rows corrupt the regression target.",
     "DROP from price model only."),
    ("Keep IQR price outliers (n=1,069, max=$540)",
     "High-end prices are real. Removing them would bias the model toward economy bookings. "
     "Apply log1p transform instead — compresses the tail without information loss.",
     "KEEP; use log1p(price)."),
    ("Keep zero-night bookings (n=78)",
     "Possibly day-use or same-arrival/departure records. Flag with is_zero_nights=1. "
     "Investigate further before dropping.",
     "FLAG, do not drop."),
    ("Keep lead_time > 365 rows (n=102)",
     "Long-advance bookings are genuine (events, weddings). Apply log1p transform.",
     "KEEP; use log1p(lead_time)."),
    ("Review calendar-invalid dates",
     "Dates like Feb 30 cannot exist. Drop or correct after manual inspection.",
     "DROP after validation — count TBD in this run."),
    ("Keep children > 5 rows (n=3)",
     "Only 3 rows; no material impact. Keep to avoid artificial filtering.",
     "KEEP."),
    ("Rare category handling (Room_Type 3 n=7, Meal Plan 3 n=5)",
     "One-hot encoding would create near-zero-variance columns. Use target encoding computed "
     "on training fold only — never on validation/test.",
     "TARGET ENCODE in pipeline."),
    ("Stratify train/val/test splits",
     "32.8% cancellation rate is a moderate imbalance. Stratified splits preserve the ratio. "
     "Evaluate class-weighted loss or SMOTE only after baseline.",
     "STRATIFY splits; evaluate class weights."),
]


def save_cleaning_decisions() -> None:
    log.info("=" * 60)
    log.info("STEP 12 — Cleaning decisions")
    log.info("=" * 60)
    tbl = pd.DataFrame(CLEANING_DECISIONS,
                       columns=["decision", "business_reasoning", "recommendation"])
    save_table(tbl, "16_cleaning_decisions")
    for dec, reason, rec in CLEANING_DECISIONS:
        log.info("  [%s]  ➜  %s", dec, rec)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("RoomRadar EDA started  — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Output tables  → %s", TABLE_DIR)
    log.info("Output figures → %s", FIG_DIR)

    df = load_and_validate()
    audit_completeness(df)
    analyse_target(df)
    audit_df = audit_suspicious(df)
    analyse_numerics(df)
    analyse_categories(df)
    analyse_cancellations(df)
    analyse_seasonality(df)
    analyse_correlations(df)
    analyse_guest_history(df)
    save_feature_definitions()
    save_cleaning_decisions()

    log.info("=" * 60)
    log.info("EDA complete — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Tables  (%d files): %s", len(list(TABLE_DIR.glob("*.csv"))), TABLE_DIR)
    log.info("Figures (%d files): %s", len(list(FIG_DIR.glob("*.png"))),  FIG_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
