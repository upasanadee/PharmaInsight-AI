"""
PharmaInsight AI — Walk-Forward Forecast Evaluation

Rolling-origin evaluation for time-series forecasting.

Supported model families:
    - Naive
    - Seasonal Naive
    - Moving Average
    - ETS
    - SARIMA
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.forecasting.baselines import (
    generate_baseline_forecasts,
)

from ml.forecasting.metrics import (
    evaluate_forecast,
)

from ml.forecasting.statistical import (
    generate_statistical_forecasts,
)


BASELINE_MODELS = {
    "Naive",
    "Seasonal Naive",
    "Moving Average",
}

STATISTICAL_MODELS = {
    "ETS",
    "SARIMA",
}


def walk_forward_baseline_evaluation(
    series: pd.Series,
    model_name: str,
    initial_train_size: int,
    horizon: int,
    step_size: int | None = None,
    seasonal_period: int = 7,
) -> pd.DataFrame:
    """
    Walk-forward evaluation for baseline and statistical models.

    No random shuffling.
    No future information leakage.
    """

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = (
        values
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if step_size is None:
        step_size = horizon

    if initial_train_size <= seasonal_period:
        raise ValueError(
            "initial_train_size must be greater than "
            "seasonal_period."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if step_size <= 0:
        raise ValueError(
            "step_size must be positive."
        )

    if initial_train_size + horizon > len(values):
        raise ValueError(
            "Not enough observations for the requested "
            "initial training size and horizon."
        )

    if (
        model_name not in BASELINE_MODELS
        and model_name not in STATISTICAL_MODELS
    ):
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    records = []

    fold = 1
    train_end = initial_train_size

    while train_end + horizon <= len(values):

        train = values.iloc[:train_end]

        actual = values.iloc[
            train_end:train_end + horizon
        ].to_numpy()

        # --------------------------------------------------
        # Generate forecast
        # --------------------------------------------------

        if model_name in BASELINE_MODELS:

            forecasts = generate_baseline_forecasts(
                training_series=train,
                horizon=horizon,
                seasonal_period=seasonal_period,
            )

            prediction = forecasts[model_name]

        elif model_name in STATISTICAL_MODELS:

            forecasts = generate_statistical_forecasts(
                training_series=train,
                horizon=horizon,
                seasonal_period=seasonal_period,
            )

            prediction = forecasts[model_name]

        else:
            raise RuntimeError(
                "Unsupported model."
            )

        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------

        metrics = evaluate_forecast(
            actual=actual,
            predicted=prediction,
            training_series=train,
            seasonality=seasonal_period,
        )

        records.append(
            {
                "fold": fold,
                "train_end": train_end,
                "horizon": horizon,
                "model": model_name,
                **metrics,
            }
        )

        fold += 1
        train_end += step_size

    return pd.DataFrame(records)


def summarize_walk_forward_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate walk-forward results by model.
    """

    metric_columns = [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    ]

    return (
        fold_results
        .groupby("model")[metric_columns]
        .agg(["mean", "std"])
    )
