from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.app.agent.state import AgentState
from backend.app.agent.tools import (
    get_category_intelligence,
    get_demand_profile,
    get_forecast,
    get_portfolio_analysis,
    get_risk_analysis,
)


CATEGORY_PATTERN = re.compile(
    r"\b(M01AB|M01AE|N02BA|N02BE|N05B|N05C|R03|R06)\b",
    re.IGNORECASE,
)


def extract_category(question: str) -> str | None:
    match = CATEGORY_PATTERN.search(question)

    if not match:
        return None

    return match.group(1).upper()


def route_question(state: AgentState) -> dict[str, Any]:
    """
    Determine which PharmaInsight analysis tool should handle the question.
    """

    question = state["question"].lower()
    category = extract_category(question)

    state_update: dict[str, Any] = {
        "category": category,
        "tool_results": [],
    }

    # Category-specific risk questions
    if category and any(
        phrase in question
        for phrase in [
            "risk",
            "danger",
            "concern",
            "attention",
            "problem",
            "issue",
        ]
    ):
        state_update["tool_results"] = [
            {
                "tool": "risk_analysis",
                "result": get_risk_analysis(category),
            }
        ]
        return state_update

    # Category-specific demand-pattern questions
    if category and any(
        phrase in question
        for phrase in [
            "regime",
            "pattern",
            "seasonal",
            "intermittent",
            "regular",
            "variable",
            "demand type",
        ]
    ):
        state_update["tool_results"] = [
            {
                "tool": "demand_profile",
                "result": get_demand_profile(category),
            }
        ]
        return state_update

    # Category-specific forecast questions
    if category and any(
        phrase in question
        for phrase in [
            "forecast",
            "predict",
            "prediction",
            "future",
            "next 30",
            "next month",
        ]
    ):
        state_update["tool_results"] = [
            {
                "tool": "forecast",
                "result": get_forecast(category),
            }
        ]
        return state_update

    # Category-specific model / general intelligence questions
    if category:
        state_update["tool_results"] = [
            {
                "tool": "category_intelligence",
                "result": get_category_intelligence(category),
            }
        ]
        return state_update

    # Portfolio-level analysis
    state_update["tool_results"] = [
        {
            "tool": "portfolio_analysis",
            "result": get_portfolio_analysis(),
        }
    ]

    return state_update


def format_number(value: Any, decimals: int = 2) -> str:
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percent(value: Any) -> str:
    try:
        number = float(value)
        sign = "+" if number >= 0 else ""
        return f"{sign}{number:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def generate_response(state: AgentState) -> dict[str, Any]:
    """
    Convert the selected tool output into an evidence-backed response.
    """

    question = state["question"].lower()
    results = state.get("tool_results", [])

    if not results:
        return {
            "answer": (
                "No PharmaInsight analysis result was available "
                "for this question."
            )
        }

    tool_name = results[0]["tool"]
    data = results[0]["result"]

    if "error" in data:
        return {"answer": data["error"]}

    # ---------------------------------------------------------
    # CATEGORY FORECAST
    # ---------------------------------------------------------

    if tool_name == "forecast":
        category = data.get("category", "Unknown")
        forecast = data.get("forecast_30d_mean")
        recent = data.get("recent_30d_mean")
        change = data.get("forecast_change_pct")
        model = data.get("model", "N/A")
        mase = data.get("MASE")

        return {
            "answer": (
                f"{category} has a 30-day forecast average of "
                f"{format_number(forecast)} units/day.\n\n"
                f"Recent 30-day demand averaged "
                f"{format_number(recent)} units/day, so the forecast "
                f"represents a {format_percent(change)} change.\n\n"
                f"The selected production model is {model}, with "
                f"a MASE of {format_number(mase, 3)}."
            )
        }

    # ---------------------------------------------------------
    # CATEGORY RISK
    # ---------------------------------------------------------

    if tool_name == "risk_analysis":
        category = data.get("category", "Unknown")
        risk = data.get("risk_level", "N/A")
        score = data.get("risk_score", "N/A")
        regime = data.get("demand_regime", "N/A")
        change = data.get("forecast_change_pct")
        mase = data.get("MASE")
        explanation = data.get("explanation")
        recommendation = data.get("recommended_action")

        answer = (
            f"{category} is currently classified as {risk} risk "
            f"with a risk score of {score}.\n\n"
            f"Demand regime: {regime}.\n"
            f"Forecast change: {format_percent(change)}.\n"
            f"MASE: {format_number(mase, 3)}."
        )

        if explanation:
            answer += f"\n\n{explanation}"

        if recommendation:
            answer += f"\n\nRecommended action: {recommendation}"

        return {"answer": answer}

    # ---------------------------------------------------------
    # CATEGORY DEMAND PROFILE
    # ---------------------------------------------------------

    if tool_name == "demand_profile":
        category = data.get("category", "Unknown")
        regime = data.get("demand_regime", "N/A")
        zero_demand = data.get("zero_demand_percentage")
        cv = data.get("coefficient_of_variation")
        trend = data.get("trend_strength")
        weekly = data.get("weekly_seasonality")
        annual = data.get("annual_seasonality")

        zero_text = (
            f"{format_number(zero_demand, 1)}%"
            if zero_demand is not None
            else "N/A"
        )

        return {
            "answer": (
                f"{category} is classified as {regime} demand.\n\n"
                f"Zero-demand observations: {zero_text}\n"
                f"Coefficient of variation: {format_number(cv, 2)}\n"
                f"Trend strength: {format_number(trend, 2)}\n"
                f"Weekly seasonality: {format_number(weekly, 2)}\n"
                f"Annual seasonality: {format_number(annual, 2)}"
            )
        }

    # ---------------------------------------------------------
    # CATEGORY GENERAL INTELLIGENCE
    # ---------------------------------------------------------

    if tool_name == "category_intelligence":
        category = data.get("category", "Unknown")
        model = data.get("model", "N/A")
        mase = data.get("MASE")
        regime = data.get("demand_regime", "N/A")
        risk = data.get("risk_level", "N/A")
        score = data.get("risk_score", "N/A")
        forecast = data.get("forecast_30d_mean")
        change = data.get("forecast_change_pct")

        return {
            "answer": (
                f"Category: {category}\n"
                f"Production model: {model}\n"
                f"MASE: {format_number(mase, 3)}\n"
                f"30-day forecast: {format_number(forecast)} units/day\n"
                f"Forecast change: {format_percent(change)}\n"
                f"Demand regime: {regime}\n"
                f"Risk level: {risk}\n"
                f"Risk score: {score}"
            )
        }

    # ---------------------------------------------------------
    # PORTFOLIO ANALYSIS
    # ---------------------------------------------------------

    if tool_name == "portfolio_analysis":
        categories = data.get("categories", [])
        high_risk = data.get("high_risk_categories", [])
        intermittent = data.get("intermittent_categories", [])

        # Largest forecast increase
        if any(
            phrase in question
            for phrase in [
                "largest forecast increase",
                "biggest forecast increase",
                "highest forecast increase",
                "largest increase",
                "biggest increase",
                "highest increase",
                "increase the most",
            ]
        ):
            top = data.get("largest_forecast_increase", {})

            return {
                "answer": (
                    f"{top.get('category')} has the largest forecast "
                    f"increase at "
                    f"{format_percent(top.get('forecast_change_pct'))} "
                    f"versus recent demand.\n\n"
                    f"Production model: {top.get('model', 'N/A')}.\n"
                    f"Demand regime: "
                    f"{top.get('demand_regime', 'N/A')}."
                )
            }

        # Largest forecast decrease
        if any(
            phrase in question
            for phrase in [
                "largest forecast decrease",
                "biggest forecast decrease",
                "highest forecast decrease",
                "largest decrease",
                "biggest decrease",
                "highest decrease",
                "decreased the most",
                "drop the most",
            ]
        ):
            bottom = data.get("largest_forecast_decrease", {})

            return {
                "answer": (
                    f"{bottom.get('category')} has the largest forecast "
                    f"decrease at "
                    f"{format_percent(bottom.get('forecast_change_pct'))} "
                    f"versus recent demand.\n\n"
                    f"Production model: {bottom.get('model', 'N/A')}.\n"
                    f"Demand regime: "
                    f"{bottom.get('demand_regime', 'N/A')}."
                )
            }

        # Lowest MASE
        if any(
            phrase in question
            for phrase in [
                "lowest mase",
                "best mase",
                "lowest error",
                "most accurate",
            ]
        ):
            best = data.get("lowest_MASE", {})

            return {
                "answer": (
                    f"{best.get('category')} has the lowest category-level "
                    f"MASE at {format_number(best.get('MASE'), 3)}.\n\n"
                    f"Production model: {best.get('model', 'N/A')}."
                )
            }

        # Intermittent categories
        if any(
            phrase in question
            for phrase in [
                "intermittent",
                "zero demand",
                "sparse demand",
            ]
        ):
            names = [
                str(item.get("category"))
                for item in intermittent
            ]

            return {
                "answer": (
                    "The categories currently classified as intermittent "
                    "demand are "
                    + ", ".join(names)
                    + "."
                )
            }

        # High-risk categories
        if any(
            phrase in question
            for phrase in [
                "high risk",
                "highest risk",
                "risky",
                "risk categories",
                "need attention",
            ]
        ):
            names = [
                str(item.get("category"))
                for item in high_risk
            ]

            return {
                "answer": (
                    "The currently high-risk categories are "
                    + ", ".join(names)
                    + "."
                )
            }

        # Portfolio overview
        changes = [
            float(item.get("forecast_change_pct", 0))
            for item in categories
        ]

        average_change = (
            sum(changes) / len(changes)
            if changes
            else 0
        )

        return {
            "answer": (
                f"PharmaInsight is currently tracking "
                f"{len(categories)} pharmaceutical categories.\n\n"
                f"Average forecast change versus recent demand: "
                f"{format_percent(average_change)}.\n"
                f"High-risk categories: {len(high_risk)}.\n"
                f"Intermittent-demand categories: {len(intermittent)}."
            )
        }

    return {
        "answer": (
            "I could not determine the appropriate PharmaInsight "
            "analysis for this question."
        )
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge(START, "route_question")
    workflow.add_edge("route_question", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


pharma_graph = build_graph()


def run_pharma_agent(question: str) -> str:
    result = pharma_graph.invoke(
        {
            "question": question,
        }
    )

    return result.get(
        "answer",
        "No answer was generated.",
    )
