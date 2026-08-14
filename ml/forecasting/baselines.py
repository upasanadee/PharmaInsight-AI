"""
PharmaInsight AI — Forecasting Baselines

Baseline models are deliberately simple.

They establish the minimum performance that
more sophisticated models must beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(
    training_series,
    horizon: int,
) -> np.ndarray:
    """
    Repeat the most recent observed value.
    """

    series = np.asarray(
        training_series,
        dtype=float,
    )

    if len(series) == 0:
        raise ValueError(
            "Training series cannot be empty."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    return np.repeat(
        series[-1],
        horizon,
    )


def seasonal_naive_forecast(
    training_series,
    horizon: int,
    seasonality: int,
) -> np.ndarray:
    """
    Repeat the values from the previous seasonal cycle.

    Example:
        seasonality=7 for daily weekly seasonality.
    """

    series = np.asarray(
        training_series,
        dtype=float,
    )

    if seasonality <= 0:
        raise ValueError(
            "seasonality must be positive."
        )

    if len(series) < seasonality:
        raise ValueError(
            "Training series is shorter than "
            "the seasonal period."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    seasonal_values = series[
        -seasonality:
    ]

    repeats = int(
        np.ceil(
            horizon / seasonality
        )
    )

    return np.tile(
        seasonal_values,
        repeats,
    )[:horizon]


def moving_average_forecast(
    training_series,
    horizon: int,
    window: int = 7,
) -> np.ndarray:
    """
    Forecast using the mean of the most recent
    observations.
    """

    series = np.asarray(
        training_series,
        dtype=float,
    )

    if len(series) == 0:
        raise ValueError(
            "Training series cannot be empty."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    window = min(
        window,
        len(series),
    )

    mean_value = np.mean(
        series[-window:]
    )

    return np.repeat(
        mean_value,
        horizon,
    )


def generate_baseline_forecasts(
    training_series,
    horizon: int,
    seasonal_period: int = 7,
) -> dict[str, np.ndarray]:
    """
    Generate all baseline forecasts.
    """

    return {
        "Naive": naive_forecast(
            training_series,
            horizon,
        ),

        "Seasonal Naive": seasonal_naive_forecast(
            training_series,
            horizon,
            seasonality=seasonal_period,
        ),

        "Moving Average": moving_average_forecast(
            training_series,
            horizon,
            window=seasonal_period,
        ),
    }