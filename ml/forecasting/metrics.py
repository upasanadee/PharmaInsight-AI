"""
PharmaInsight AI — Forecasting Metrics

Evaluation metrics designed for pharmaceutical demand forecasting.

Includes:
- MAE
- RMSE
- MAPE
- sMAPE
- WAPE
- MASE
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_inputs(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual and predicted must have the same shape."
        )

    if actual.size == 0:
        raise ValueError(
            "actual and predicted cannot be empty."
        )

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    return actual[mask], predicted[mask]


def mae(actual, predicted) -> float:
    """Mean Absolute Error."""

    actual, predicted = _validate_inputs(
        actual,
        predicted,
    )

    return float(
        np.mean(
            np.abs(actual - predicted)
        )
    )


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error."""

    actual, predicted = _validate_inputs(
        actual,
        predicted,
    )

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def mape(
    actual,
    predicted,
    epsilon: float = 1e-8,
) -> float:
    """
    Mean Absolute Percentage Error.

    Zero actual-demand observations are excluded because
    percentage error is undefined at zero.
    """

    actual, predicted = _validate_inputs(
        actual,
        predicted,
    )

    mask = np.abs(actual) > epsilon

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100
    )


def smape(
    actual,
    predicted,
    epsilon: float = 1e-8,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    More robust than MAPE when demand contains small values.
    """

    actual, predicted = _validate_inputs(
        actual,
        predicted,
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    mask = denominator > epsilon

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            2
            * np.abs(
                actual[mask] - predicted[mask]
            )
            / denominator[mask]
        )
        * 100
    )


def wape(
    actual,
    predicted,
    epsilon: float = 1e-8,
) -> float:
    """
    Weighted Absolute Percentage Error.

        WAPE = sum(|actual - predicted|)
               / sum(|actual|)
    """

    actual, predicted = _validate_inputs(
        actual,
        predicted,
    )

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator <= epsilon:
        return np.nan

    return float(
        np.sum(
            np.abs(actual - predicted)
        )
        / denominator
        * 100
    )


def mase(
    actual,
    predicted,
    training_series,
    seasonality: int = 1,
    epsilon: float = 1e-8,
) -> float:
    """
    Mean Absolute Scaled Error.

    Scales forecast error against a naive
    in-sample forecast.

    MASE < 1:
        Better than naive benchmark.

    MASE > 1:
        Worse than naive benchmark.
    """

    actual, predicted = _validate_inputs(
        actual,
        predicted,
    )

    training = np.asarray(
        training_series,
        dtype=float,
    )

    training = training[
        np.isfinite(training)
    ]

    if len(training) <= seasonality:
        return np.nan

    naive_errors = np.abs(
        training[seasonality:]
        - training[:-seasonality]
    )

    scale = np.mean(
        naive_errors
    )

    if scale <= epsilon:
        return np.nan

    forecast_error = np.mean(
        np.abs(
            actual - predicted
        )
    )

    return float(
        forecast_error / scale
    )


def evaluate_forecast(
    actual,
    predicted,
    training_series=None,
    seasonality: int = 1,
) -> dict[str, float]:
    """
    Calculate the complete forecasting metric suite.
    """

    results = {
        "MAE": mae(
            actual,
            predicted,
        ),
        "RMSE": rmse(
            actual,
            predicted,
        ),
        "MAPE": mape(
            actual,
            predicted,
        ),
        "sMAPE": smape(
            actual,
            predicted,
        ),
        "WAPE": wape(
            actual,
            predicted,
        ),
    }

    if training_series is not None:

        results["MASE"] = mase(
            actual,
            predicted,
            training_series,
            seasonality=seasonality,
        )
    else:
        results["MASE"] = np.nan

    return results


def evaluate_forecast_dataframe(
    actual,
    predicted,
    training_series=None,
    seasonality: int = 1,
) -> pd.DataFrame:
    """
    Return forecast metrics as a one-row DataFrame.
    """

    results = evaluate_forecast(
        actual=actual,
        predicted=predicted,
        training_series=training_series,
        seasonality=seasonality,
    )

    return pd.DataFrame(
        [results]
    )