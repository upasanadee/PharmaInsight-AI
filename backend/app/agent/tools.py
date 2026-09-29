from __future__ import annotations

from typing import Any

from backend.app.services.intelligence_service import (
    get_all_categories,
    get_category_context,
    get_high_risk_categories,
    get_intermittent_categories,
)


def get_category_intelligence(category: str) -> dict[str, Any]:
    """Return the complete intelligence profile for one category."""
    context = get_category_context(category)

    if not context:
        return {
            "error": f"No intelligence data found for category {category.upper()}."
        }

    return context


def get_forecast(category: str) -> dict[str, Any]:
    """Return forecast and validation information for one category."""
    context = get_category_context(category)

    if not context:
        return {
            "error": f"No forecast data found for category {category.upper()}."
        }

    return {
        "category": context.get("category"),
        "model": context.get("model"),
        "recent_30d_mean": context.get("recent_30d_mean"),
        "forecast_30d_mean": context.get("forecast_30d_mean"),
        "forecast_change_pct": context.get("forecast_change_pct"),
        "MASE": context.get("MASE"),
    }


def get_risk_analysis(category: str) -> dict[str, Any]:
    """Return forecast risk and evidence for one category."""
    context = get_category_context(category)

    if not context:
        return {
            "error": f"No risk data found for category {category.upper()}."
        }

    return {
        "category": context.get("category"),
        "risk_level": context.get("risk_level"),
        "risk_score": context.get("risk_score"),
        "demand_regime": context.get("demand_regime"),
        "explanation": context.get("explanation"),
        "recommended_action": context.get("recommended_action"),
        "risk_factors": context.get("risk_factors", []),
        "forecast_change_pct": context.get("forecast_change_pct"),
        "MASE": context.get("MASE"),
    }


def get_demand_profile(category: str) -> dict[str, Any]:
    """Return demand-pattern information for one category."""
    context = get_category_context(category)

    if not context:
        return {
            "error": f"No demand profile found for category {category.upper()}."
        }

    return {
        "category": context.get("category"),
        "demand_regime": context.get("demand_regime"),
        "zero_demand_percentage": context.get("zero_demand_percentage"),
        "coefficient_of_variation": context.get(
            "coefficient_of_variation"
        ),
        "trend_strength": context.get("trend_strength"),
        "weekly_seasonality": context.get("weekly_seasonality"),
        "annual_seasonality": context.get("annual_seasonality"),
    }


def get_portfolio_analysis() -> dict[str, Any]:
    """Return portfolio-level forecasting intelligence."""
    categories = get_all_categories()
    high_risk = get_high_risk_categories()
    intermittent = get_intermittent_categories()

    if not categories:
        return {
            "error": "No portfolio intelligence data is available."
        }

    ranked_increase = sorted(
        categories,
        key=lambda item: float(
            item.get("forecast_change_pct", 0)
        ),
        reverse=True,
    )

    ranked_decrease = sorted(
        categories,
        key=lambda item: float(
            item.get("forecast_change_pct", 0)
        ),
    )

    ranked_mase = sorted(
        categories,
        key=lambda item: float(
            item.get("MASE", 999)
        ),
    )

    return {
        "category_count": len(categories),
        "high_risk_categories": high_risk,
        "intermittent_categories": intermittent,
        "largest_forecast_increase": ranked_increase[0],
        "largest_forecast_decrease": ranked_decrease[0],
        "lowest_MASE": ranked_mase[0],
        "categories": categories,
    }
