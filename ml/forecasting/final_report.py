"""
PharmaInsight AI — Final Business Forecast Report

Combines:
1. Final model comparison
2. Production forecasts
3. Forecast sanity diagnostics

Produces a business-facing forecast report for all drug categories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.loader import load_dataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPORT_DIR = Path("reports")
DIAGNOSTIC_DIR = REPORT_DIR / "forecast_diagnostics"

FINAL_COMPARISON = REPORT_DIR / "final_model_comparison.csv"
SANITY_CHECK = DIAGNOSTIC_DIR / "forecast_sanity_check.csv"

OUTPUT_FILE = REPORT_DIR / "final_business_forecast_report.csv"


TARGET_COLUMNS = [
    "M01AB",
    "M01AE",
    "N02BA",
    "N02BE",
    "N05B",
    "N05C",
    "R03",
    "R06",
]


# ---------------------------------------------------------------------
# Load production forecasts
# ---------------------------------------------------------------------

def load_production_forecasts(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the current production forecasts used by the
    forecast sanity check.

    These forecasts correspond to the production forecast period
    immediately following the final observed date.
    """

    from ml.forecasting.production_forecast import (
        generate_production_forecasts,
    )

    forecasts = generate_production_forecasts(
        dataframe=dataframe,
        horizon=30,
    )

    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError(
            "generate_production_forecasts() must return a "
            "pandas DataFrame."
        )

    return forecasts


# ---------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------

def classify_status(
    category: str,
    forecast_vs_recent_ratio: float,
) -> str:
    """
    Assign a business-facing forecast status.

    N05C is explicitly marked as intermittent demand because its
    demand profile contains a very large proportion of zero values.
    """

    if category == "N05C":
        return "INTERMITTENT_DEMAND_REVIEW"

    if not np.isfinite(forecast_vs_recent_ratio):
        return "REVIEW"

    change = abs(forecast_vs_recent_ratio - 1.0)

    if change <= 0.15:
        return "OK"

    if change <= 0.30:
        return "MODERATE_CHANGE"

    return "HIGH_CHANGE_REVIEW"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print()
    print("=" * 80)
    print("PHARMAINSIGHT AI — FINAL BUSINESS FORECAST REPORT")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    dataframe = load_dataset("daily")

    dataframe["datum"] = pd.to_datetime(
        dataframe["datum"]
    )

    dataframe = dataframe.sort_values(
        "datum"
    ).reset_index(drop=True)

    historical_start = dataframe["datum"].min()
    historical_end = dataframe["datum"].max()

    print()
    print(
        f"Historical period: "
        f"{historical_start} → {historical_end}"
    )

    # ---------------------------------------------------------------
    # Load model comparison
    # ---------------------------------------------------------------

    if not FINAL_COMPARISON.exists():
        raise FileNotFoundError(
            f"Missing final model comparison:\n"
            f"{FINAL_COMPARISON}"
        )

    model_comparison = pd.read_csv(
        FINAL_COMPARISON
    )

    required_model_columns = {
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    }

    missing = (
        required_model_columns
        - set(model_comparison.columns)
    )

    if missing:
        raise ValueError(
            "Final model comparison is missing columns: "
            f"{sorted(missing)}"
        )

    # Select best model according to MASE.
    best_models = (
        model_comparison
        .sort_values(
            ["category", "MASE"],
            ascending=[True, True],
        )
        .groupby(
            "category",
            as_index=False,
        )
        .first()
    )

    # ---------------------------------------------------------------
    # Load sanity diagnostics
    # ---------------------------------------------------------------

    if not SANITY_CHECK.exists():
        raise FileNotFoundError(
            f"Missing forecast sanity check:\n"
            f"{SANITY_CHECK}"
        )

    diagnostics = pd.read_csv(
        SANITY_CHECK
    )

    required_diagnostic_columns = {
        "category",
        "historical_mean",
        "historical_std",
        "recent_30d_mean",
        "forecast_30d_mean",
        "forecast_min",
        "forecast_max",
        "forecast_vs_recent_ratio",
    }

    missing = (
        required_diagnostic_columns
        - set(diagnostics.columns)
    )

    if missing:
        raise ValueError(
            "Forecast diagnostics are missing columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------------
    # Merge model performance and forecast diagnostics
    # ---------------------------------------------------------------

    report = best_models.merge(
        diagnostics,
        on="category",
        how="inner",
    )

    if len(report) != len(TARGET_COLUMNS):
        missing_categories = sorted(
            set(TARGET_COLUMNS)
            - set(report["category"])
        )

        raise ValueError(
            "Could not construct a complete report. "
            f"Missing categories: {missing_categories}"
        )

    # ---------------------------------------------------------------
    # Add business metrics
    # ---------------------------------------------------------------

    report["forecast_change_pct"] = (
        (
            report["forecast_vs_recent_ratio"]
            - 1.0
        )
        * 100.0
    )

    report["status"] = [
        classify_status(
            category,
            ratio,
        )
        for category, ratio in zip(
            report["category"],
            report["forecast_vs_recent_ratio"],
        )
    ]

    # ---------------------------------------------------------------
    # Add interpretation
    # ---------------------------------------------------------------

    interpretations = []

    for _, row in report.iterrows():

        category = row["category"]
        ratio = row["forecast_vs_recent_ratio"]

        if category == "N05C":
            interpretation = (
                "Intermittent low-volume demand; "
                "forecast requires review."
            )

        elif ratio > 1.30:
            interpretation = (
                "Forecast materially above recent demand."
            )

        elif ratio > 1.15:
            interpretation = (
                "Forecast moderately above recent demand."
            )

        elif ratio < 0.70:
            interpretation = (
                "Forecast materially below recent demand."
            )

        elif ratio < 0.85:
            interpretation = (
                "Forecast moderately below recent demand."
            )

        else:
            interpretation = (
                "Forecast broadly aligned with recent demand."
            )

        interpretations.append(
            interpretation
        )

    report["business_interpretation"] = (
        interpretations
    )

    # ---------------------------------------------------------------
    # Forecast period
    # ---------------------------------------------------------------

    forecast_start = (
        historical_end
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        historical_end
        + pd.Timedelta(days=30)
    )

    report["historical_start"] = (
        historical_start
    )

    report["historical_end"] = (
        historical_end
    )

    report["forecast_start"] = (
        forecast_start
    )

    report["forecast_end"] = (
        forecast_end
    )

    # ---------------------------------------------------------------
    # Select final columns
    # ---------------------------------------------------------------

    final_columns = [
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
        "historical_mean",
        "historical_std",
        "recent_30d_mean",
        "forecast_30d_mean",
        "forecast_min",
        "forecast_max",
        "forecast_vs_recent_ratio",
        "forecast_change_pct",
        "status",
        "business_interpretation",
        "historical_start",
        "historical_end",
        "forecast_start",
        "forecast_end",
    ]

    report = report[
        final_columns
    ]

    report = report.sort_values(
        "category"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL BUSINESS FORECAST REPORT")
    print("=" * 80)

    display_columns = [
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
        "recent_30d_mean",
        "forecast_30d_mean",
        "forecast_change_pct",
        "status",
    ]

    print(
        report[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("BUSINESS INTERPRETATION")
    print("=" * 80)

    for _, row in report.iterrows():

        print()
        print(f"{row['category']}")
        print(f"  Selected model : {row['model']}")
        print(
            f"  Recent demand  : "
            f"{row['recent_30d_mean']:.3f}"
        )
        print(
            f"  Forecast       : "
            f"{row['forecast_30d_mean']:.3f}"
        )
        print(
            f"  Change         : "
            f"{row['forecast_change_pct']:+.2f}%"
        )
        print(
            f"  Status         : "
            f"{row['status']}"
        )
        print(
            f"  Interpretation : "
            f"{row['business_interpretation']}"
        )

    print()
    print("=" * 80)
    print("REPORT SAVED")
    print("=" * 80)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
