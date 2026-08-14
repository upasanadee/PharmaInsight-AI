"""
PharmaInsight AI — Seasonality & Forecastability Analysis

Provides quantitative measures of:

- Trend strength
- Weekly seasonality
- Annual seasonality
- Demand intermittency
- Demand volatility

The functions are designed to operate on daily pharmaceutical
demand time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL


def _validate_series(
    series: pd.Series,
    minimum_length: int,
) -> pd.Series:
    """
    Validate and clean a univariate demand series.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    cleaned = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(cleaned) < minimum_length:
        raise ValueError(
            f"At least {minimum_length} observations are required; "
            f"received {len(cleaned)}."
        )

    return cleaned.astype(float)


def _stl_strength(
    series: pd.Series,
    period: int,
    component: str,
) -> float:
    """
    Calculate STL-based trend or seasonal strength.

    Strength is defined as:

        max(0, 1 - Var(remainder) /
                 Var(remainder + component))

    Values closer to 1 indicate a stronger component.
    """

    minimum_length = max(
        period * 2 + 1,
        30,
    )

    cleaned = _validate_series(
        series,
        minimum_length=minimum_length,
    )

    # STL requires a meaningful time series.
    if cleaned.nunique() <= 1:
        return 0.0

    try:
        result = STL(
            cleaned,
            period=period,
            robust=True,
        ).fit()

        if component == "trend":
            component_values = result.trend

        elif component == "seasonal":
            component_values = result.seasonal

        else:
            raise ValueError(
                "component must be either 'trend' or 'seasonal'."
            )

        remainder = result.resid

        denominator = np.var(
            remainder + component_values
        )

        if denominator <= 1e-12:
            return 0.0

        strength = (
            1.0
            - np.var(remainder)
            / denominator
        )

        return float(
            np.clip(strength, 0.0, 1.0)
        )

    except Exception:
        return np.nan


def calculate_trend_strength(
    series: pd.Series,
    period: int = 365,
) -> float:
    """
    Calculate long-term trend strength using STL.

    Parameters
    ----------
    series:
        Daily demand series.

    period:
        Seasonal period used for decomposition.
        Default is 365 for annual daily data.

    Returns
    -------
    float
        Trend strength between 0 and 1.
    """

    return _stl_strength(
        series,
        period=period,
        component="trend",
    )


def calculate_weekly_seasonality(
    series: pd.Series,
) -> float:
    """
    Calculate weekly seasonality strength.

    Uses a 7-day STL decomposition.

    Returns
    -------
    float
        Weekly seasonality strength between 0 and 1.
    """

    return _stl_strength(
        series,
        period=7,
        component="seasonal",
    )


def calculate_annual_seasonality(
    series: pd.Series,
) -> float:
    """
    Calculate annual seasonality strength.

    Uses a 365-day STL decomposition.

    Returns
    -------
    float
        Annual seasonality strength between 0 and 1.
    """

    return _stl_strength(
        series,
        period=365,
        component="seasonal",
    )


def calculate_zero_demand_percentage(
    series: pd.Series,
) -> float:
    """
    Percentage of observations with zero demand.
    """

    cleaned = _validate_series(
        series,
        minimum_length=1,
    )

    return float(
        (cleaned == 0).mean() * 100
    )


def calculate_coefficient_of_variation(
    series: pd.Series,
) -> float:
    """
    Calculate coefficient of variation:

        CV = standard deviation / mean
    """

    cleaned = _validate_series(
        series,
        minimum_length=2,
    )

    mean = cleaned.mean()

    if abs(mean) <= 1e-12:
        return np.inf

    return float(
        cleaned.std() / mean
    )


def classify_demand_regime(
    coefficient_of_variation: float,
    zero_demand_percentage: float,
    weekly_seasonality: float,
    annual_seasonality: float,
    trend_strength: float,
) -> str:
    """
    Classify a demand series into a high-level forecasting regime.

    Classification is intentionally conservative. It is used to guide
    model benchmarking, not to determine the final production model.

    Regimes
    -------
    Intermittent:
        Large proportion of zero-demand observations.

    Highly Variable:
        High relative variability without dominant seasonality.

    Seasonal:
        Strong weekly or annual seasonal structure.

    Trending:
        Strong long-term trend.

    Regular:
        No dominant difficulty signal.
    """

    if zero_demand_percentage >= 20:
        return "Intermittent"

    if coefficient_of_variation >= 1.0:
        return "Highly Variable"

    if (
        weekly_seasonality >= 0.30
        or annual_seasonality >= 0.30
    ):
        return "Seasonal"

    if trend_strength >= 0.30:
        return "Trending"

    return "Regular"


def build_forecastability_profile(
    dataframe: pd.DataFrame,
    target_columns: list[str],
    date_column: str = "datum",
) -> pd.DataFrame:
    """
    Build a quantitative forecastability profile for all demand categories.

    Parameters
    ----------
    dataframe:
        Daily demand dataframe.

    target_columns:
        Pharmaceutical demand columns.

    date_column:
        Date column used to validate chronological data.

    Returns
    -------
    pandas.DataFrame
        One row per demand category.
    """

    if date_column not in dataframe.columns:
        raise ValueError(
            f"Date column '{date_column}' not found."
        )

    missing_columns = [
        column
        for column in target_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing target columns: {missing_columns}"
        )

    working = dataframe.copy()

    working[date_column] = pd.to_datetime(
        working[date_column]
    )

    working = working.sort_values(
        date_column
    )

    records = []

    for category in target_columns:

        series = working[category]

        cv = calculate_coefficient_of_variation(
            series
        )

        zero_pct = calculate_zero_demand_percentage(
            series
        )

        trend_strength = calculate_trend_strength(
            series
        )

        weekly_strength = calculate_weekly_seasonality(
            series
        )

        annual_strength = calculate_annual_seasonality(
            series
        )

        lag_1 = series.autocorr(
            lag=1
        )

        lag_7 = series.autocorr(
            lag=7
        )

        lag_14 = series.autocorr(
            lag=14
        )

        lag_30 = series.autocorr(
            lag=30
        )

        lag_365 = series.autocorr(
            lag=365
        )

        regime = classify_demand_regime(
            coefficient_of_variation=cv,
            zero_demand_percentage=zero_pct,
            weekly_seasonality=weekly_strength,
            annual_seasonality=annual_strength,
            trend_strength=trend_strength,
        )

        records.append(
            {
                "category": category,
                "mean": series.mean(),
                "std": series.std(),
                "coefficient_of_variation": cv,
                "skewness": series.skew(),
                "kurtosis": series.kurt(),
                "zero_demand_pct": zero_pct,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_30": lag_30,
                "lag_365": lag_365,
                "trend_strength": trend_strength,
                "weekly_seasonality_strength": weekly_strength,
                "annual_seasonality_strength": annual_strength,
                "demand_regime": regime,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "category"
        )
        .reset_index(drop=True)
    )