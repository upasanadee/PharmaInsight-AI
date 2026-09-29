"""
PharmaInsight AI — Demand Intelligence Profile

Builds a quantitative demand profile for each pharmaceutical category
using the existing seasonality and forecastability analysis.
"""

from pathlib import Path

from ml.analytics.seasonality import build_forecastability_profile
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

OUTPUT_DIR = Path("reports/intelligence")
OUTPUT_FILE = OUTPUT_DIR / "demand_profile.csv"


def main() -> None:
    print("=" * 78)
    print("PHARMAINSIGHT AI — DEMAND INTELLIGENCE PROFILE")
    print("=" * 78)

    dataframe = load_dataset("daily")

    profile = build_forecastability_profile(
        dataframe=dataframe,
        target_columns=TARGET_COLUMNS,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Demand regimes:")
    print(
        profile[
            [
                "category",
                "coefficient_of_variation",
                "zero_demand_pct",
                "trend_strength",
                "weekly_seasonality_strength",
                "annual_seasonality_strength",
                "demand_regime",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
