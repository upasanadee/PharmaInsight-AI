"""
PharmaInsight AI — Dashboard Data Builder

Consolidates validated forecasting outputs into dashboard-ready files.

Inputs:
    reports/production_30_day_forecast.csv
    reports/final_business_forecast_report.csv

Outputs:
    reports/dashboard/category_summary.csv
    reports/dashboard/forecast_daily.csv
    reports/dashboard/dashboard_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORTS_DIR = Path("reports")
OUTPUT_DIR = REPORTS_DIR / "dashboard"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:

    print("=" * 80)
    print("PHARMAINSIGHT AI — DASHBOARD DATA BUILDER")
    print("=" * 80)

    production_path = REPORTS_DIR / "production_30_day_forecast.csv"
    business_path = REPORTS_DIR / "final_business_forecast_report.csv"

    production = pd.read_csv(production_path)
    business = pd.read_csv(business_path)

    production["datum"] = pd.to_datetime(production["datum"])

    # --------------------------------------------------------------
    # Validate production forecasts
    # --------------------------------------------------------------

    expected_categories = sorted(business["category"].unique())

    if len(production) != len(expected_categories) * 30:
        raise ValueError(
            "Unexpected production forecast size: "
            f"{len(production)} rows. "
            f"Expected {len(expected_categories) * 30}."
        )

    if production["forecast"].isna().any():
        raise ValueError("Production forecasts contain NaN values.")

    if (production["forecast"] < 0).any():
        raise ValueError("Production forecasts contain negative values.")

    # --------------------------------------------------------------
    # Category summary
    # --------------------------------------------------------------

    summary_columns = [
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

    category_summary = business[summary_columns].copy()

    category_summary = category_summary.sort_values(
        "category"
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # Daily forecast table
    # --------------------------------------------------------------

    forecast_daily = production[
        ["datum", "category", "model", "forecast"]
    ].copy()

    forecast_daily = forecast_daily.sort_values(
        ["category", "datum"]
    ).reset_index(drop=True)

    forecast_daily["forecast_day"] = (
        forecast_daily.groupby("category").cumcount() + 1
    )

    # --------------------------------------------------------------
    # Dashboard-level KPIs
    # --------------------------------------------------------------

    total_categories = len(category_summary)

    flagged_categories = int(
        (category_summary["status"] != "OK").sum()
    )

    total_forecast_demand = float(
        forecast_daily["forecast"].sum()
    )

    total_recent_demand = float(
        category_summary["recent_30d_mean"].sum() * 30
    )

    overall_change_pct = (
        (total_forecast_demand - total_recent_demand)
        / total_recent_demand
        * 100
        if total_recent_demand > 0
        else float("nan")
    )

    best_mase_category = category_summary.loc[
        category_summary["MASE"].idxmin(),
        "category",
    ]

    best_mase_model = category_summary.loc[
        category_summary["MASE"].idxmin(),
        "model",
    ]

    model_counts = (
        category_summary["model"]
        .value_counts()
        .to_dict()
    )

    dashboard_summary = pd.DataFrame(
        [
            {
                "total_categories": total_categories,
                "flagged_categories": flagged_categories,
                "forecast_horizon_days": 30,
                "total_forecast_demand": total_forecast_demand,
                "total_recent_30d_demand": total_recent_demand,
                "overall_change_pct": overall_change_pct,
                "best_mase_category": best_mase_category,
                "best_mase_model": best_mase_model,
                "model_counts": "; ".join(
                    f"{model}: {count}"
                    for model, count in sorted(model_counts.items())
                ),
            }
        ]
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    category_path = OUTPUT_DIR / "category_summary.csv"
    daily_path = OUTPUT_DIR / "forecast_daily.csv"
    dashboard_path = OUTPUT_DIR / "dashboard_summary.csv"

    category_summary.to_csv(
        category_path,
        index=False,
    )

    forecast_daily.to_csv(
        daily_path,
        index=False,
    )

    dashboard_summary.to_csv(
        dashboard_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print()
    print("Category summary:")
    print(
        category_summary[
            [
                "category",
                "model",
                "recent_30d_mean",
                "forecast_30d_mean",
                "forecast_change_pct",
                "MASE",
                "status",
            ]
        ].to_string(index=False)
    )

    print()
    print("Dashboard KPIs:")
    print(dashboard_summary.to_string(index=False))

    print()
    print("Saved:")
    print(category_path)
    print(daily_path)
    print(dashboard_path)


if __name__ == "__main__":
    main()
