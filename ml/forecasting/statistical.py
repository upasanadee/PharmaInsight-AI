"""
PharmaInsight AI — Statistical Forecasting Models

Statistical forecasting models used as an intermediate layer
between simple baselines and machine-learning models.

Models:
    - ETS / Holt-Winters Exponential Smoothing
    - SARIMA

Important:
    Models are fitted only on historical observations.
    No future observations are used during fitting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _prepare_series(
    series: pd.Series,
) -> pd.Series:
    """Clean and validate a univariate time series."""

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(values) < 30:
        raise ValueError(
            "At least 30 observations are required."
        )

    return values.astype(float).reset_index(
        drop=True
    )


def forecast_ets(
    training_series: pd.Series,
    horizon: int,
    seasonal_period: int = 7,
) -> np.ndarray:
    """
    Generate forecasts using Holt-Winters ETS.

    Components:
        trend      = additive
        seasonality = additive
        seasonal_period = 7 by default

    Parameters
    ----------
    training_series:
        Historical demand.

    horizon:
        Number of future observations.

    seasonal_period:
        Seasonal cycle.

    Returns
    -------
    np.ndarray
        Forecast values.
    """

    series = _prepare_series(
        training_series
    )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if len(series) < (
        seasonal_period * 2
    ):
        raise ValueError(
            "Not enough observations for "
            "the requested seasonal period."
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated",
        )

        fitted = model.fit(
            optimized=True
        )

    forecast = fitted.forecast(
        horizon
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    # Demand cannot be negative.
    forecast = np.maximum(
        forecast,
        0.0,
    )

    return forecast


def forecast_sarima(
    training_series: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (
        1,
        1,
        1,
    ),
    seasonal_order: tuple[int, int, int, int] = (
        1,
        0,
        1,
        7,
    ),
) -> np.ndarray:
    """
    Generate forecasts using SARIMA.

    Default configuration:

        ARIMA order       = (1,1,1)
        Seasonal order    = (1,0,1,7)

    Parameters
    ----------
    training_series:
        Historical demand.

    horizon:
        Number of future observations.

    order:
        Non-seasonal ARIMA parameters.

    seasonal_order:
        Seasonal ARIMA parameters.

    Returns
    -------
    np.ndarray
        Forecast values.
    """

    series = _prepare_series(
        training_series
    )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted = model.fit(
            disp=False
        )

    forecast = fitted.forecast(
        steps=horizon
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    forecast = np.maximum(
        forecast,
        0.0,
    )

    return forecast


def generate_statistical_forecasts(
    training_series: pd.Series,
    horizon: int,
    seasonal_period: int = 7,
) -> dict[str, np.ndarray]:
    """
    Generate all statistical forecasts.

    Returns
    -------
    dict
        Model name -> forecast.
    """

    return {
        "ETS": forecast_ets(
            training_series=training_series,
            horizon=horizon,
            seasonal_period=seasonal_period,
        ),
        "SARIMA": forecast_sarima(
            training_series=training_series,
            horizon=horizon,
        ),
    }
