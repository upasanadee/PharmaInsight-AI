from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_FILE = PROJECT_ROOT / "reports" / "intelligence" / "forecast_risk.csv"
PROFILE_FILE = PROJECT_ROOT / "reports" / "intelligence" / "demand_profile.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "final_business_forecast_report.csv"
EXPLANATION_FILE = (
    PROJECT_ROOT / "reports" / "intelligence" / "forecast_explanations.csv"
)

OUTPUT_FILE = PROJECT_ROOT / "reports" / "intelligence" / "recommendations.csv"


def load_data() -> pd.DataFrame:
    risk = pd.read_csv(RISK_FILE)
    profile = pd.read_csv(PROFILE_FILE)
    report = pd.read_csv(REPORT_FILE)
    explanations = pd.read_csv(EXPLANATION_FILE)

    columns = [
        "category",
        "risk_level",
        "risk_score",
        "demand_regime",
        "forecast_change_pct",
        "MASE",
    ]

    dataframe = (
        risk[["category", "risk_level", "risk_score"]]
        .merge(
            profile[["category", "demand_regime"]],
            on="category",
            how="left",
        )
        .merge(
            report[["category", "forecast_change_pct", "MASE"]],
            on="category",
            how="left",
        )
        .merge(
            explanations[["category", "explanation"]],
            on="category",
            how="left",
        )
    )

    return dataframe[columns + ["explanation"]]


def build_recommendation(row: pd.Series) -> tuple[str, str, str]:
    category = row["category"]
    risk_level = str(row["risk_level"]).upper()
    regime = str(row["demand_regime"])
    change = float(row["forecast_change_pct"])
    mase = float(row["MASE"])

    if category == "N05C":
        return (
            "HIGH",
            "Review recent sales and inventory before replenishment decisions.",
            (
                "Demand is intermittent and highly variable, with a large "
                "forecast increase relative to recent demand."
            ),
        )

    if category == "R03":
        return (
            "HIGH",
            "Review demand history and inventory levels before planning replenishment.",
            (
                "Demand is intermittent and highly variable, increasing "
                "uncertainty around the forecast."
            ),
        )

    if category == "R06":
        return (
            "MEDIUM",
            "Monitor demand and inventory because the forecast is materially below recent demand.",
            (
                f"The forecast is {abs(change):.1f}% below recent demand "
                f"and the forecast error remains elevated (MASE {mase:.2f})."
            ),
        )

    if risk_level == "HIGH":
        return (
            "HIGH",
            "Investigate recent demand and inventory before making planning decisions.",
            (
                f"The category has elevated forecast risk with a MASE of "
                f"{mase:.2f}."
            ),
        )

    if regime == "Seasonal":
        return (
            "LOW",
            "Continue routine monitoring with attention to seasonal demand patterns.",
            "The category exhibits a recurring seasonal demand pattern.",
        )

    if abs(change) >= 15:
        direction = "above" if change > 0 else "below"

        return (
            "LOW",
            f"Monitor demand because the forecast is {abs(change):.1f}% {direction} recent demand.",
            "The forecast shows a noticeable movement relative to recent demand.",
        )

    return (
        "LOW",
        "Continue routine demand and forecast monitoring.",
        "The forecast is broadly aligned with recent demand and has low forecast risk.",
    )


def build_recommendations() -> pd.DataFrame:
    dataframe = load_data()

    recommendations = dataframe.apply(
        build_recommendation,
        axis=1,
        result_type="expand",
    )

    recommendations.columns = [
        "priority",
        "recommendation",
        "reason",
    ]

    dataframe = pd.concat(
        [
            dataframe[
                [
                    "category",
                    "risk_level",
                    "risk_score",
                    "demand_regime",
                    "forecast_change_pct",
                    "MASE",
                ]
            ],
            recommendations,
        ],
        axis=1,
    )

    dataframe = dataframe.sort_values(
        by=["risk_score", "MASE"],
        ascending=[False, False],
    )

    return dataframe


def main() -> None:
    dataframe = build_recommendations()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved recommendations to: {OUTPUT_FILE}")
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()