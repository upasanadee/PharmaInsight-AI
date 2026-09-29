from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from backend.app.agent.graph import run_pharma_agent

from backend.app.services.intelligence_service import (
    get_all_categories,
    get_category_context,
    get_high_risk_categories,
    get_intermittent_categories,
    load_intelligence_context,
)

router = APIRouter()


class AssistantRequest(BaseModel):
    question: str


class AssistantResponse(BaseModel):
    answer: str
    source: str


CATEGORY_PATTERN = re.compile(
    r"\b(M01AB|M01AE|N02BA|N02BE|N05B|N05C|R03|R06)\b",
    re.IGNORECASE,
)


def extract_category(question: str) -> str | None:
    match = CATEGORY_PATTERN.search(question)

    if not match:
        return None

    return match.group(1).upper()


def clean_number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def clean_percent(value: Any) -> str:
    try:
        number = float(value)
        sign = "+" if number >= 0 else ""
        return f"{sign}{number:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_category_context(context: dict[str, Any]) -> str:
    category = context.get("category", "Unknown")
    model = context.get("model", "N/A")

    mase = context.get("MASE")
    forecast_change = context.get("forecast_change_pct")
    demand_regime = context.get("demand_regime")
    risk_level = context.get("risk_level")
    risk_score = context.get("risk_score")

    recent = context.get("recent_30d_mean")
    forecast = context.get("forecast_30d_mean")

    explanation = context.get("explanation")
    recommendation = context.get("recommended_action")

    lines = [
        f"Category: {category}",
        f"Production model: {model}",
        f"MASE: {clean_number(mase, 3)}",
        f"Recent 30-day demand: {clean_number(recent)}",
        f"30-day forecast demand: {clean_number(forecast)}",
        f"Forecast change: {clean_percent(forecast_change)}",
        f"Demand regime: {demand_regime or 'N/A'}",
        f"Risk level: {risk_level or 'N/A'}",
        f"Risk score: {risk_score if risk_score is not None else 'N/A'}",
    ]

    if explanation:
        lines.append(f"Explanation: {explanation}")

    if recommendation:
        lines.append(f"Recommended action: {recommendation}")

    return "\n".join(lines)


def category_answer(question: str, category: str) -> str:
    context = get_category_context(category)

    if not context:
        return (
            f"I couldn't find intelligence data for {category}. "
            "Available categories are M01AB, M01AE, N02BA, "
            "N02BE, N05B, N05C, R03 and R06."
        )

    q = question.lower()

    model = context.get("model", "N/A")
    mase = context.get("MASE")
    change = context.get("forecast_change_pct")
    regime = context.get("demand_regime", "N/A")
    risk = context.get("risk_level", "N/A")
    score = context.get("risk_score", "N/A")
    recent = context.get("recent_30d_mean")
    forecast = context.get("forecast_30d_mean")
    explanation = context.get("explanation", "")
    recommendation = context.get("recommended_action", "")

    # Risk / safety questions
    if any(
        word in q
        for word in [
            "risk",
            "danger",
            "concern",
            "attention",
            "problem",
            "issue",
        ]
    ):
        return (
            f"{category} is currently classified as {risk} risk "
            f"with a risk score of {score}.\n\n"
            f"Demand regime: {regime}.\n"
            f"Forecast change: {clean_percent(change)}.\n"
            f"MASE: {clean_number(mase, 3)}.\n\n"
            f"{explanation}\n\n"
            f"Recommended action: {recommendation}"
        )

    # Model questions
    if any(
        word in q
        for word in [
            "model",
            "algorithm",
            "method",
            "which model",
            "what model",
        ]
    ):
        return (
            f"{category} uses {model} as its production forecasting "
            f"model.\n\n"
            f"Its validation MASE is {clean_number(mase, 3)}. "
            f"The model was selected from the category-level "
            "forecasting benchmark."
        )

    # Accuracy questions
    if any(
        word in q
        for word in [
            "accurate",
            "accuracy",
            "error",
            "mase",
            "performance",
            "reliable",
        ]
    ):
        return (
            f"{category} has a MASE of {clean_number(mase, 3)} "
            f"using {model}.\n\n"
            f"Forecast change versus recent demand is "
            f"{clean_percent(change)}.\n"
            f"Demand regime: {regime}.\n"
            f"Current risk level: {risk}."
        )

    # Forecast questions
    if any(
        word in q
        for word in [
            "forecast",
            "predict",
            "prediction",
            "future",
            "next 30",
            "next month",
            "demand",
        ]
    ):
        return (
            f"{category} has a 30-day forecast average of "
            f"{clean_number(forecast)} units/day.\n\n"
            f"Recent 30-day demand averaged "
            f"{clean_number(recent)} units/day, so the forecast "
            f"represents a {clean_percent(change)} change.\n\n"
            f"The selected production model is {model}, with "
            f"a MASE of {clean_number(mase, 3)}."
        )

    # Demand regime questions
    if any(
        word in q
        for word in [
            "regime",
            "pattern",
            "seasonal",
            "intermittent",
            "regular",
            "variable",
            "demand type",
        ]
    ):
        return (
            f"{category} is classified as {regime} demand.\n\n"
            f"The production model is {model}, with a MASE of "
            f"{clean_number(mase, 3)}.\n"
            f"Current forecast change: {clean_percent(change)}."
        )

    # General category question
    return format_category_context(context)


def portfolio_answer(question: str) -> str:
    q = question.lower()

    context = load_intelligence_context()

    # ---------------------------------------------------------
    # HIGH-RISK CATEGORIES
    # ---------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "high risk",
            "highest risk",
            "risky",
            "risk categories",
            "risk category",
        ]
    ):
        high_risk = get_high_risk_categories()

        if not high_risk:
            return "No high-risk categories are currently identified."

        names = [
            str(item.get("category"))
            for item in high_risk
        ]

        details = []

        for item in high_risk:
            details.append(
                f"{item.get('category')}: "
                f"risk {item.get('risk_level')}, "
                f"score {item.get('risk_score')}"
            )

        return (
            f"{len(names)} categories currently have elevated "
            "forecast risk: "
            + ", ".join(names)
            + ".\n\n"
            + "\n".join(details)
        )

    # ---------------------------------------------------------
    # INTERMITTENT DEMAND
    # ---------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "intermittent",
            "zero demand",
            "sparse demand",
            "sporadic demand",
        ]
    ):
        intermittent = get_intermittent_categories()

        if not intermittent:
            return "No categories are currently classified as intermittent demand."

        names = [
            str(item.get("category"))
            for item in intermittent
        ]

        return (
            "The categories currently classified as intermittent "
            "demand are "
            + ", ".join(names)
            + ".\n\n"
            "These categories contain substantial periods of zero "
            "or sparse demand, so their forecasts require more "
            "careful interpretation."
        )

    # ---------------------------------------------------------
    # LARGEST FORECAST INCREASE / DECREASE
    # ---------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "largest forecast increase",
            "biggest forecast increase",
            "highest forecast increase",
            "largest increase",
            "biggest increase",
            "highest increase",
            "increase the most",
            "increased the most",
            "forecast increase",
        ]
    ):
        categories = get_all_categories()

        if not categories:
            return "No category forecast data is available."

        ranked = sorted(
            categories,
            key=lambda x: float(
                x.get("forecast_change_pct", 0)
            ),
            reverse=True,
        )

        top = ranked[0]

        return (
            f"{top.get('category')} has the largest forecast "
            f"increase at "
            f"{clean_percent(top.get('forecast_change_pct'))} "
            "versus recent demand.\n\n"
            f"Production model: {top.get('model')}.\n"
            f"Demand regime: {top.get('demand_regime', 'N/A')}."
        )

    if any(
        phrase in q
        for phrase in [
            "largest decrease",
            "biggest decrease",
            "highest decrease",
            "decreased the most",
            "drop the most",
        ]
    ):
        categories = get_all_categories()

        if not categories:
            return "No category forecast data is available."

        ranked = sorted(
            categories,
            key=lambda x: float(
                x.get("forecast_change_pct", 0)
            ),
        )

        bottom = ranked[0]

        return (
            f"{bottom.get('category')} has the largest forecast "
            f"decrease at "
            f"{clean_percent(bottom.get('forecast_change_pct'))} "
            "versus recent demand.\n\n"
            f"Production model: {bottom.get('model')}.\n"
            f"Demand regime: {bottom.get('demand_regime', 'N/A')}."
        )

    # ---------------------------------------------------------
    # BEST / LOWEST MASE
    # ---------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "lowest mase",
            "best mase",
            "best performing",
            "best performing category",
            "most accurate category",
            "lowest error",
        ]
    ):
        categories = get_all_categories()

        if not categories:
            return "No category performance data is available."

        ranked = sorted(
            categories,
            key=lambda x: float(
                x.get("MASE", 999)
            ),
        )

        best = ranked[0]

        return (
            f"{best.get('category')} has the lowest category-level "
            f"MASE at {clean_number(best.get('MASE'), 3)}.\n\n"
            f"Production model: {best.get('model')}.\n"
            f"Demand regime: {best.get('demand_regime', 'N/A')}.\n"
            f"Forecast change: "
            f"{clean_percent(best.get('forecast_change_pct'))}."
        )

    # ---------------------------------------------------------
    # SEASONAL CATEGORIES
    # ---------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "seasonal categories",
            "which categories are seasonal",
            "seasonal demand",
        ]
    ):
        categories = get_all_categories()

        seasonal = [
            item
            for item in categories
            if str(item.get("demand_regime", "")).lower()
            == "seasonal"
        ]

        if not seasonal:
            return "No categories are currently classified as seasonal."

        names = [
            str(item.get("category"))
            for item in seasonal
        ]

        return (
            "The categories currently classified as seasonal "
            "demand are "
            + ", ".join(names)
            + "."
        )

    # ---------------------------------------------------------
    # PORTFOLIO OVERVIEW
    # ---------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "overview",
            "portfolio",
            "overall",
            "what is happening",
            "what's happening",
            "summarize",
            "summary",
            "how are we doing",
        ]
    ):
        categories = get_all_categories()
        high_risk = get_high_risk_categories()
        intermittent = get_intermittent_categories()

        if not categories:
            return "No intelligence data is currently available."

        changes = [
            float(item.get("forecast_change_pct", 0))
            for item in categories
        ]

        average_change = sum(changes) / len(changes)

        return (
            f"PharmaInsight is currently tracking "
            f"{len(categories)} pharmaceutical categories.\n\n"
            f"Average forecast change versus recent demand: "
            f"{clean_percent(average_change)}.\n"
            f"High-risk categories: "
            f"{len(high_risk)}.\n"
            f"Intermittent-demand categories: "
            f"{len(intermittent)}.\n\n"
            "The main categories requiring closer review are "
            + (
                ", ".join(
                    str(item.get("category"))
                    for item in high_risk
                )
                if high_risk
                else "none currently flagged"
            )
            + "."
        )

    # ---------------------------------------------------------
    # GENERAL GUIDANCE
    # ---------------------------------------------------------

    return (
        "I can answer questions about PharmaInsight's actual "
        "forecasting data. Try questions such as:\n\n"
        "• What is the forecast for N02BE?\n"
        "• Why is N05C high risk?\n"
        "• Which categories need attention?\n"
        "• Which categories have intermittent demand?\n"
        "• Which category has the largest forecast increase?\n"
        "• Which category has the lowest MASE?\n"
        "• Which categories are seasonal?\n"
        "• What model is used for M01AE?\n"
        "• How accurate is R06?\n"
        "• Give me an overview of the current portfolio."
    )


@router.post(
    "/assistant",
    response_model=AssistantResponse,
)
def assistant(request: AssistantRequest) -> AssistantResponse:
    question = request.question.strip()

    if not question:
        return AssistantResponse(
            answer="Please enter a question about the PharmaInsight forecasting data.",
            source="PharmaInsight intelligence reports",
        )

    try:
        answer = run_pharma_agent(question)

    except Exception as exc:
        return AssistantResponse(
            answer=(
                "The PharmaInsight analysis could not be completed. "
                f"Error: {exc}"
            ),
            source="PharmaInsight intelligence reports",
        )

    return AssistantResponse(
        answer=answer,
        source="PharmaInsight LangGraph intelligence workflow",
    )