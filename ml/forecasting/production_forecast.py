"""
PharmaInsight AI — Production Forecasting Pipeline

Uses the best model identified by the final benchmark for each
pharmaceutical category and generates a 30-day future forecast.

Model selection has already been completed using chronological
validation/testing.

Production training:
    Uses the complete available historical dataset.

Forecast horizon:
    30 days.

Negative predictions are clipped to zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.loader import load_dataset
from ml.forecasting.features import build_supervised_dataset
from ml.forecasting.ml_models import (
    create_xgboost_model,
    create_lightgbm_model,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

TARGET_COLUMNS = [
    "M01AB",
    "M01AE",
    "N02BA",
    "N02BE",
    "N05B",
    "N05C",
    "R03",
    "R06",
]

FORECAST_HORIZON = 30

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    REPORT_DIR / "production_30_day_forecast.csv"
)


# ------------------------------------------------------------------
# Selected models from final benchmark
# ------------------------------------------------------------------

BEST_MODELS = {
    "M01AB": "LightGBM",
    "M01AE": "SARIMA",
    "N02BA": "SARIMA",
    "N02BE": "LightGBM",
    "N05B": "SARIMA",
    "N05C": "LightGBM",
    "R03": "SARIMA",
    "R06": "SARIMA",
}


# ------------------------------------------------------------------
# ML chronological training
# ------------------------------------------------------------------

def chronological_train_ml(
    dataframe: pd.DataFrame,
    category: str,
    model_name: str,
) -> np.ndarray:
    """
    Train the selected ML model on all available supervised
    historical observations and generate a recursive 30-day forecast.

    The features module creates historical lag/rolling features.
    Future predictions are generated recursively.
    """

    X, y = build_supervised_dataset(
        dataframe=dataframe,
        target_column=category,
        horizon=1,
    )

    if model_name == "LightGBM":
        model = create_lightgbm_model(
            random_state=42,
        )

    elif model_name == "XGBoost":
        model = create_xgboost_model(
            random_state=42,
        )

    else:
        raise ValueError(
            f"Unsupported ML model: {model_name}"
        )

    # Train using all available supervised observations.
    model.fit(
        X,
        y,
    )

    # --------------------------------------------------------------
    # Recursive future forecasting
    # --------------------------------------------------------------

    history = dataframe[
        ["datum", category]
    ].copy()

    history["datum"] = pd.to_datetime(
        history["datum"]
    )

    history = history.sort_values(
        "datum"
    ).reset_index(drop=True)

    predictions = []

    for _ in range(FORECAST_HORIZON):

        next_date = (
            history["datum"].iloc[-1]
            + pd.Timedelta(days=1)
        )

        # Build a temporary dataframe containing the current
        # historical values plus the date to be forecast.
        temp = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "datum": [next_date],
                        category: [np.nan],
                    }
                ),
            ],
            ignore_index=True,
        )

        # Generate features using historical observations.
        from ml.forecasting.features import (
            create_forecasting_features,
        )

        feature_df = create_forecasting_features(
            temp,
            target_column=category,
        )

        latest_features = feature_df.iloc[
            [-1]
        ].copy()

        # Remove target/date columns.
        latest_features = latest_features.drop(
            columns=[
                "datum",
                category,
            ],
            errors="ignore",
        )

        # The production feature row must not contain NaN/inf.
        latest_features = latest_features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if latest_features.isna().any().any():
            raise ValueError(
                f"Invalid future features generated for "
                f"{category}."
            )

        prediction = float(
            model.predict(
                latest_features
            )[0]
        )

        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(prediction)

        # Add prediction to history so the next forecast step
        # can use it as a lagged observation.
        history.loc[len(history)] = [
            next_date,
            prediction,
        ]

    return np.asarray(
        predictions,
        dtype=float,
    )


# ------------------------------------------------------------------
# ETS
# ------------------------------------------------------------------

def forecast_ets(
    series: pd.Series,
) -> np.ndarray:
    """Generate a 30-day ETS forecast."""

    series = pd.Series(
        series,
        dtype=float,
    ).dropna()

    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )

    fitted = model.fit(
        optimized=True
    )

    forecast = fitted.forecast(
        FORECAST_HORIZON
    )

    return np.maximum(
        np.asarray(forecast, dtype=float),
        0.0,
    )


# ------------------------------------------------------------------
# SARIMA
# ------------------------------------------------------------------

def forecast_sarima(
    series: pd.Series,
) -> np.ndarray:
    """Generate a 30-day SARIMA forecast."""

    series = pd.Series(
        series,
        dtype=float,
    ).dropna()

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False
    )

    forecast = fitted.forecast(
        FORECAST_HORIZON
    )

    return np.maximum(
        np.asarray(forecast, dtype=float),
        0.0,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    print()
    print("=" * 78)
    print(
        "PHARMAINSIGHT AI — PRODUCTION FORECASTING"
    )
    print("=" * 78)

    dataframe = load_dataset(
        "daily"
    )

    dataframe["datum"] = pd.to_datetime(
        dataframe["datum"]
    )

    dataframe = dataframe.sort_values(
        "datum"
    ).reset_index(drop=True)

    last_date = dataframe["datum"].iloc[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D",
    )

    print()
    print(
        f"Historical observations: "
        f"{len(dataframe):,}"
    )

    print(
        f"Historical period: "
        f"{dataframe['datum'].min()} → "
        f"{last_date}"
    )

    print(
        f"Forecast period: "
        f"{future_dates[0]} → "
        f"{future_dates[-1]}"
    )

    forecasts = pd.DataFrame(
        {
            "datum": future_dates,
        }
    )

    selected_models = {}

    # --------------------------------------------------------------
    # Generate category-specific forecasts
    # --------------------------------------------------------------

    for category in TARGET_COLUMNS:

        model_name = BEST_MODELS[
            category
        ]

        selected_models[
            category
        ] = model_name

        print()
        print(
            f"Forecasting {category} "
            f"using {model_name}..."
        )

        series = dataframe[
            category
        ].astype(float)

        if model_name in {
            "LightGBM",
            "XGBoost",
        }:

            prediction = chronological_train_ml(
                dataframe=dataframe,
                category=category,
                model_name=model_name,
            )

        elif model_name == "SARIMA":

            prediction = forecast_sarima(
                series
            )

        elif model_name == "ETS":

            prediction = forecast_ets(
                series
            )

        else:

            raise ValueError(
                f"Unknown production model: "
                f"{model_name}"
            )

        forecasts[
            category
        ] = prediction

        print(
            f"  Minimum forecast: "
            f"{prediction.min():.3f}"
        )

        print(
            f"  Maximum forecast: "
            f"{prediction.max():.3f}"
        )

    # --------------------------------------------------------------
    # Long-format business output
    # --------------------------------------------------------------

    long_forecast = forecasts.melt(
        id_vars="datum",
        value_vars=TARGET_COLUMNS,
        var_name="category",
        value_name="forecast",
    )

    long_forecast[
        "model"
    ] = long_forecast[
        "category"
    ].map(
        selected_models
    )

    long_forecast = long_forecast[
        [
            "datum",
            "category",
            "model",
            "forecast",
        ]
    ]

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    long_forecast.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 78)
    print("30-DAY FORECAST SUMMARY")
    print("=" * 78)

    print(
        long_forecast.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
