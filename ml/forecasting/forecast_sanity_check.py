"""
PharmaInsight AI — Production Forecast Sanity Check

Checks the 30-day production forecasts against recent and
historical demand.

Outputs:
    - forecast summary statistics
    - historical mean/std
    - recent 30-day mean
    - forecast mean
    - forecast vs recent demand ratio
    - min/max forecast
    - simple anomaly flags
    - visualization plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.preprocessing.loader import load_dataset


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

FORECAST_FILE = Path(
    "reports/production_30_day_forecast.csv"
)

OUTPUT_DIR = Path(
    "reports/forecast_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main() -> None:

    print()
    print("=" * 78)
    print("PHARMAINSIGHT AI — FORECAST SANITY CHECK")
    print("=" * 78)

    # --------------------------------------------------------------
    # Load historical data
    # --------------------------------------------------------------

    historical = load_dataset("daily")

    historical["datum"] = pd.to_datetime(
        historical["datum"]
    )

    historical = historical.sort_values(
        "datum"
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # Load production forecast
    # --------------------------------------------------------------

    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["datum"],
    )

    print()
    print(
        f"Historical period: "
        f"{historical['datum'].min().date()} → "
        f"{historical['datum'].max().date()}"
    )

    print(
        f"Forecast period: "
        f"{forecast['datum'].min().date()} → "
        f"{forecast['datum'].max().date()}"
    )

    # --------------------------------------------------------------
    # Diagnostic table
    # --------------------------------------------------------------

    diagnostics = []

    for category in TARGET_COLUMNS:

        hist = historical[
            category
        ].astype(float).dropna()

        pred = forecast.loc[
            forecast["category"] == category,
            "forecast",
        ].astype(float)

        recent_30 = hist.tail(30)

        historical_mean = hist.mean()
        historical_std = hist.std()

        recent_mean = recent_30.mean()
        forecast_mean = pred.mean()

        forecast_min = pred.min()
        forecast_max = pred.max()

        if recent_mean > 0:
            forecast_ratio = (
                forecast_mean / recent_mean
            )
        else:
            forecast_ratio = np.nan

        # Flag unusually large change relative to
        # the most recent 30 days.
        if np.isfinite(forecast_ratio):

            if forecast_ratio > 1.50:
                flag = "HIGH_INCREASE"

            elif forecast_ratio < 0.50:
                flag = "HIGH_DECREASE"

            else:
                flag = "OK"

        else:
            flag = "CHECK"

        diagnostics.append(
            {
                "category": category,
                "historical_mean": historical_mean,
                "historical_std": historical_std,
                "recent_30d_mean": recent_mean,
                "forecast_30d_mean": forecast_mean,
                "forecast_min": forecast_min,
                "forecast_max": forecast_max,
                "forecast_vs_recent_ratio": forecast_ratio,
                "status": flag,
            }
        )

    diagnostics = pd.DataFrame(
        diagnostics
    )

    # --------------------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("FORECAST DIAGNOSTICS")
    print("=" * 78)

    print(
        diagnostics.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # --------------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------------

    diagnostics_file = (
        OUTPUT_DIR
        / "forecast_sanity_check.csv"
    )

    diagnostics.to_csv(
        diagnostics_file,
        index=False,
    )

    # --------------------------------------------------------------
    # Generate one plot per category
    # --------------------------------------------------------------

    for category in TARGET_COLUMNS:

        hist = historical[
            ["datum", category]
        ].tail(180).copy()

        pred = forecast[
            forecast["category"] == category
        ].copy()

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            hist["datum"],
            hist[category],
            label="Historical demand",
        )

        plt.plot(
            pred["datum"],
            pred["forecast"],
            label="30-day forecast",
        )

        plt.axvline(
            historical["datum"].max(),
            linestyle="--",
            label="Forecast start",
        )

        plt.title(
            f"{category} — Historical Demand and "
            f"30-Day Forecast"
        )

        plt.xlabel("Date")
        plt.ylabel("Demand")

        plt.legend()
        plt.grid(alpha=0.25)

        plt.tight_layout()

        output_plot = (
            OUTPUT_DIR
            / f"{category}_forecast.png"
        )

        plt.savefig(
            output_plot,
            dpi=200,
        )

        plt.close()

    # --------------------------------------------------------------
    # Final output
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("SANITY CHECK COMPLETE")
    print("=" * 78)

    print()
    print("Diagnostics:")
    print(diagnostics_file)

    print()
    print("Plots:")
    for category in TARGET_COLUMNS:
        print(
            OUTPUT_DIR
            / f"{category}_forecast.png"
        )


if __name__ == "__main__":
    main()
