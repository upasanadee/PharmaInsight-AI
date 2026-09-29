"""
PharmaInsight AI — Forecast Explainability

Generates deterministic, evidence-backed explanations for forecast
risk signals. No generative model is used here; every explanation
is derived directly from the computed analytics.
"""

from pathlib import Path

import pandas as pd


REPORTS_DIR = Path("reports")
INTELLIGENCE_DIR = REPORTS_DIR / "intelligence"

RISK_FILE = INTELLIGENCE_DIR / "forecast_risk.csv"
BUSINESS_FILE = REPORTS_DIR / "final_business_forecast_report.csv"

OUTPUT_FILE = INTELLIGENCE_DIR / "forecast_explanations.csv"


def format_change(value: float) -> str:
    """Format forecast movement for human-readable explanations."""
    if value >= 0:
        return f"{value:.1f}% above"
    return f"{abs(value):.1f}% below"


def build_risk_factors(row: pd.Series) -> list[dict]:
    """
    Build transparent risk factors using the same deterministic
    thresholds as the forecast risk engine.
    """

    factors = []

    regime = row["demand_regime"]
    cv = float(row["coefficient_of_variation"])
    mase = float(row["MASE"])
    change = float(row["forecast_change_pct"])
    zero_pct = float(row["zero_demand_pct"])

    if regime == "Intermittent":
        factors.append(
            {
                "factor": "Intermittent demand",
                "points": 2,
                "evidence": (
                    f"{zero_pct:.1f}% of observations "
                    "have zero demand"
                ),
            }
        )

    if cv >= 1.0:
        factors.append(
            {
                "factor": "High demand variability",
                "points": 2,
                "evidence": f"CV = {cv:.2f}",
            }
        )

    if mase >= 1.0:
        factors.append(
            {
                "factor": "High forecast error",
                "points": 2,
                "evidence": f"MASE = {mase:.2f}",
            }
        )
    elif mase >= 0.85:
        factors.append(
            {
                "factor": "Elevated forecast error",
                "points": 1,
                "evidence": f"MASE = {mase:.2f}",
            }
        )

    if abs(change) >= 30:
        factors.append(
            {
                "factor": "Large forecast movement",
                "points": 2,
                "evidence": (
                    f"forecast is {abs(change):.1f}% "
                    f"{'above' if change >= 0 else 'below'} "
                    "recent demand"
                ),
            }
        )
    elif abs(change) >= 15:
        factors.append(
            {
                "factor": "Moderate forecast movement",
                "points": 1,
                "evidence": (
                    f"forecast is {abs(change):.1f}% "
                    f"{'above' if change >= 0 else 'below'} "
                    "recent demand"
                ),
            }
        )

    return factors


def build_explanation(row: pd.Series) -> tuple[str, str, str]:
    """
    Build a headline, explanation, and recommended action
    from deterministic forecast signals.
    """

    category = row["category"]
    regime = row["demand_regime"]
    risk_level = row["risk_level"]

    cv = float(row["coefficient_of_variation"])
    zero_pct = float(row["zero_demand_pct"])
    mase = float(row["MASE"])
    change = float(row["forecast_change_pct"])

    change_text = format_change(change)

    # ----------------------------------------------------------
    # HIGH-RISK CATEGORIES
    # ----------------------------------------------------------

    if risk_level == "HIGH":

        headline = f"{category} requires forecast review"

        reasons = []

        if regime == "Intermittent":
            reasons.append(
                f"demand is intermittent, with "
                f"{zero_pct:.1f}% zero-demand observations"
            )

        if cv >= 1.0:
            reasons.append(
                f"demand variability is high "
                f"(CV {cv:.2f})"
            )

        if mase >= 0.85:
            reasons.append(
                f"forecast error is relatively elevated "
                f"(MASE {mase:.2f})"
            )

        if abs(change) >= 15:
            reasons.append(
                f"the 30-day forecast is "
                f"{change_text} recent demand"
            )

        explanation = (
            f"{category} is classified as {regime.lower()} demand. "
            + ". ".join(
                reason.capitalize()
                for reason in reasons
            )
            + "."
        )

        action = (
            "Review recent sales and inventory conditions before "
            "using the forecast for replenishment decisions."
        )

        return headline, explanation, action

    # ----------------------------------------------------------
    # MEDIUM / ATTENTION SIGNALS
    # ----------------------------------------------------------

    if risk_level == "MEDIUM":

        headline = f"{category} warrants monitoring"

        explanation = (
            f"{category} shows {regime.lower()} demand with "
            f"a MASE of {mase:.2f}. "
            f"The 30-day forecast is {change_text} recent demand."
        )

        action = (
            "Monitor the category and investigate the forecast "
            "if the demand movement persists."
        )

        return headline, explanation, action

    # ----------------------------------------------------------
    # LOW-RISK CATEGORIES
    # ----------------------------------------------------------

    headline = f"{category} has stable forecast signals"

    if abs(change) < 5:
        movement = "The forecast remains close to recent demand."
    else:
        movement = (
            f"The forecast is {change_text} recent demand."
        )

    explanation = (
        f"{category} has {regime.lower()} demand and a "
        f"MASE of {mase:.2f}. {movement}"
    )

    action = (
        "Continue routine monitoring; no immediate forecast "
        "review is indicated by the current risk signals."
    )

    return headline, explanation, action


def main() -> None:

    print("=" * 78)
    print("PHARMAINSIGHT AI — FORECAST EXPLAINABILITY")
    print("=" * 78)

    risk = pd.read_csv(RISK_FILE)

    business = pd.read_csv(
        BUSINESS_FILE,
        usecols=[
            "category",
            "recent_30d_mean",
            "forecast_30d_mean",
        ],
    )

    data = risk.merge(
        business,
        on="category",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(risk):
        raise ValueError(
            "Explainability merge did not preserve all risk-profile rows."
        )

    explanations = []

    for _, row in data.iterrows():

        headline, explanation, action = build_explanation(row)

        factors = build_risk_factors(row)

        explanations.append(
            {
                "category": row["category"],
                "risk_level": row["risk_level"],
                "risk_score": row["risk_score"],
                "demand_regime": row["demand_regime"],
                "risk_factors": factors,
                "headline": headline,
                "explanation": explanation,
                "recommended_action": action,
            }
        )

    result = (
        pd.DataFrame(explanations)
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

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Forecast explanations:")
    print(
        result[
            [
                "category",
                "risk_level",
                "risk_score",
                "risk_factors",
                "headline",
                "explanation",
                "recommended_action",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Rows generated: {len(result)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
