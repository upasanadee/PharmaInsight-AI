"""
PharmaInsight AI — Forecast Risk Intelligence

Combines demand-regime characteristics with forecast accuracy
and forecast movement to identify categories requiring attention.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports")
INTELLIGENCE_DIR = REPORTS_DIR / "intelligence"

BUSINESS_REPORT = REPORTS_DIR / "final_business_forecast_report.csv"
DEMAND_PROFILE = INTELLIGENCE_DIR / "demand_profile.csv"
OUTPUT_FILE = INTELLIGENCE_DIR / "forecast_risk.csv"


def calculate_risk(row: pd.Series) -> tuple[int, str, str]:
    """
    Calculate a transparent risk score from deterministic signals.

    Score components:
        +2 intermittent demand
        +2 high coefficient of variation
        +2 high MASE
        +1 moderate MASE
        +2 large forecast movement
        +1 moderate forecast movement
    """

    score = 0
    reasons = []

    if row["demand_regime"] == "Intermittent":
        score += 2
        reasons.append("intermittent demand")

    if row["coefficient_of_variation"] >= 1.0:
        score += 2
        reasons.append("high demand variability")

    if row["MASE"] >= 1.0:
        score += 2
        reasons.append("forecast error above naive benchmark")
    elif row["MASE"] >= 0.85:
        score += 1
        reasons.append("moderate forecast error")

    abs_change = abs(row["forecast_change_pct"])

    if abs_change >= 30:
        score += 2
        reasons.append("large forecast movement")
    elif abs_change >= 15:
        score += 1
        reasons.append("moderate forecast movement")

    if score >= 5:
        level = "HIGH"
    elif score >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    reason_text = (
        "; ".join(reasons)
        if reasons
        else "stable demand and forecast signals"
    )

    return score, level, reason_text


def main() -> None:
    print("=" * 78)
    print("PHARMAINSIGHT AI — FORECAST RISK INTELLIGENCE")
    print("=" * 78)

    business = pd.read_csv(BUSINESS_REPORT)
    profile = pd.read_csv(DEMAND_PROFILE)

    merged = business.merge(
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
        ],
        on="category",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(business):
        raise ValueError(
            "Risk profile merge did not preserve all business-report rows."
        )

    results = []

    for _, row in merged.iterrows():
        score, level, reasons = calculate_risk(row)

        results.append(
            {
                "category": row["category"],
                "model": row["model"],
                "demand_regime": row["demand_regime"],
                "MASE": row["MASE"],
                "forecast_change_pct": row["forecast_change_pct"],
                "coefficient_of_variation": row[
                    "coefficient_of_variation"
                ],
                "zero_demand_pct": row["zero_demand_pct"],
                "trend_strength": row["trend_strength"],
                "weekly_seasonality_strength": row[
                    "weekly_seasonality_strength"
                ],
                "annual_seasonality_strength": row[
                    "annual_seasonality_strength"
                ],
                "risk_score": score,
                "risk_level": level,
                "risk_reasons": reasons,
            }
        )

    risk_profile = (
        pd.DataFrame(results)
        .sort_values(
            ["risk_score", "category"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    INTELLIGENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_profile.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Forecast risk profile:")
    print(
        risk_profile[
            [
                "category",
                "demand_regime",
                "MASE",
                "forecast_change_pct",
                "risk_score",
                "risk_level",
                "risk_reasons",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
