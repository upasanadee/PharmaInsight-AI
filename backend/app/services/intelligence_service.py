from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORTS_DIR = PROJECT_ROOT / "reports"
INTELLIGENCE_DIR = REPORTS_DIR / "intelligence"
DASHBOARD_DIR = REPORTS_DIR / "dashboard"


BUSINESS_FILE = (
    REPORTS_DIR / "final_business_forecast_report.csv"
)

CATEGORY_FILE = (
    DASHBOARD_DIR / "category_summary.csv"
)

DEMAND_PROFILE_FILE = (
    INTELLIGENCE_DIR / "demand_profile.csv"
)

RISK_FILE = (
    INTELLIGENCE_DIR / "forecast_risk.csv"
)

EXPLANATIONS_FILE = (
    INTELLIGENCE_DIR / "forecast_explanations.csv"
)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV report and fail clearly if it is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    return pd.read_csv(path)


def _normalise_category_column(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise category values for reliable joins."""

    dataframe = dataframe.copy()

    if "category" in dataframe.columns:
        dataframe["category"] = (
            dataframe["category"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    return dataframe


def _parse_risk_factors(value) -> list:
    """Convert stored risk-factor strings back to Python lists."""

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        return []

    except (ValueError, SyntaxError):
        return []


def _is_missing_scalar(value) -> bool:
    """
    Safely determine whether a value is missing.

    Handles normal scalar NaN values without attempting to
    evaluate lists/arrays as booleans.
    """

    if value is None:
        return True

    if isinstance(value, (list, tuple, dict)):
        return False

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        return False

    except (TypeError, ValueError):
        return False


def load_intelligence_context() -> list[dict]:
    """
    Build one complete intelligence record per category.

    The dashboard/business report provides:
        - model
        - forecast metrics
        - recent demand
        - forecast demand
        - forecast movement
        - status

    The demand profile provides:
        - demand regime
        - variability
        - zero-demand percentage
        - trend/seasonality

    The risk report provides:
        - risk score
        - risk level
        - risk reasons

    The explanation report provides:
        - headline
        - explanation
        - recommended action
        - risk factors
    """

    # --------------------------------------------------------------
    # Load source reports
    # --------------------------------------------------------------

    business = _normalise_category_column(
        _read_csv(BUSINESS_FILE)
    )

    category_summary = _normalise_category_column(
        _read_csv(CATEGORY_FILE)
    )

    demand_profile = _normalise_category_column(
        _read_csv(DEMAND_PROFILE_FILE)
    )

    risk = _normalise_category_column(
        _read_csv(RISK_FILE)
    )

    explanations = _normalise_category_column(
        _read_csv(EXPLANATIONS_FILE)
    )

    # --------------------------------------------------------------
    # Start with dashboard/business data
    # --------------------------------------------------------------

    # category_summary is the canonical source for forecast
    # values exposed by the dashboard.
    base = category_summary.copy()

    # If the dashboard file is unavailable/incomplete, use
    # the business report as the fallback.
    if base.empty:
        base = business.copy()

    # --------------------------------------------------------------
    # Add demand-profile intelligence
    # --------------------------------------------------------------

    demand_columns = [
        column
        for column in [
            "category",
            "demand_regime",
            "coefficient_of_variation",
            "zero_demand_percentage",
            "zero_demand_pct",
            "trend_strength",
            "weekly_seasonality_strength",
            "annual_seasonality_strength",
        ]
        if column in demand_profile.columns
    ]

    if demand_columns:
        base = base.merge(
            demand_profile[demand_columns],
            on="category",
            how="left",
        )

    # --------------------------------------------------------------
    # Add forecast-risk intelligence
    # --------------------------------------------------------------

    risk_columns = [
        column
        for column in [
            "category",
            "risk_score",
            "risk_level",
            "risk_reasons",
        ]
        if column in risk.columns
    ]

    if risk_columns:
        base = base.merge(
            risk[risk_columns],
            on="category",
            how="left",
        )

    # --------------------------------------------------------------
    # Add explainability intelligence
    # --------------------------------------------------------------

    explanation_columns = [
        column
        for column in [
            "category",
            "headline",
            "explanation",
            "recommended_action",
            "risk_factors",
        ]
        if column in explanations.columns
    ]

    if explanation_columns:
        base = base.merge(
            explanations[explanation_columns],
            on="category",
            how="left",
        )

    # --------------------------------------------------------------
    # Clean duplicate columns created by merges
    # --------------------------------------------------------------

    for column in [
        "model",
        "MASE",
        "forecast_change_pct",
    ]:
        x_column = f"{column}_x"
        y_column = f"{column}_y"

        if x_column in base.columns:

            if column not in base.columns:
                base[column] = base[x_column]

            else:
                base[column] = base[column].fillna(
                    base[x_column]
                )

        if y_column in base.columns:

            if column not in base.columns:
                base[column] = base[y_column]

            else:
                base[column] = base[column].fillna(
                    base[y_column]
                )

    # --------------------------------------------------------------
    # Build clean records
    # --------------------------------------------------------------

    records = []

    for _, row in base.iterrows():

        record = row.to_dict()

        # ----------------------------------------------------------
        # Normalise category
        # ----------------------------------------------------------

        record["category"] = str(
            record.get("category", "")
        ).strip().upper()

        # ----------------------------------------------------------
        # Parse risk factors
        # ----------------------------------------------------------

        record["risk_factors"] = _parse_risk_factors(
            record.get("risk_factors")
        )

        # ----------------------------------------------------------
        # Safely convert scalar NaN values to None
        # ----------------------------------------------------------

        for key, value in list(record.items()):

            if _is_missing_scalar(value):
                record[key] = None

        records.append(record)

    return records


def get_category_context(
    category: str,
) -> dict | None:
    """Return complete intelligence context for one category."""

    category = category.strip().upper()

    records = load_intelligence_context()

    for record in records:

        if record.get("category") == category:
            return record

    return None


def get_high_risk_categories() -> list[dict]:
    """Return categories classified as HIGH risk."""

    records = load_intelligence_context()

    return [
        record
        for record in records
        if str(
            record.get("risk_level", "")
        ).upper()
        == "HIGH"
    ]


def get_intermittent_categories() -> list[dict]:
    """Return categories classified as intermittent demand."""

    records = load_intelligence_context()

    return [
        record
        for record in records
        if str(
            record.get("demand_regime", "")
        ).lower()
        == "intermittent"
    ]


def get_all_categories() -> list[dict]:
    """Return complete intelligence records."""

    return load_intelligence_context()