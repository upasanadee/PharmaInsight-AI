"""
PharmaInsight AI — Forecasting Feature Engineering

Creates leakage-safe features for machine-learning forecasting.

Feature groups
--------------
1. Calendar features
2. Trend features
3. Lag features
4. Rolling statistics
5. Cyclical seasonal features

IMPORTANT
---------
All lag and rolling features use only historical observations.
The current target value is never used to construct its own features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LAGS = [
    1,
    2,
    3,
    7,
    14,
    21,
    28,
    30,
    60,
    90,
    365,
]


DEFAULT_ROLLING_WINDOWS = [
    7,
    14,
    30,
    60,
    90,
]


def create_forecasting_features(
    dataframe: pd.DataFrame,
    target_column: str,
    date_column: str = "datum",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Create leakage-safe forecasting features.

    Parameters
    ----------
    dataframe:
        Input time-series dataframe.

    target_column:
        Demand column to forecast.

    date_column:
        Timestamp column.

    lags:
        Lag periods to create.

    rolling_windows:
        Rolling windows to create.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """

    if date_column not in dataframe.columns:
        raise ValueError(
            f"Missing date column: {date_column}"
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    if lags is None:
        lags = DEFAULT_LAGS

    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS

    data = dataframe[
        [date_column, target_column]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data = (
        data
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    data["year"] = (
        data[date_column].dt.year
    )

    data["month"] = (
        data[date_column].dt.month
    )

    data["quarter"] = (
        data[date_column].dt.quarter
    )

    data["day_of_week"] = (
        data[date_column].dt.dayofweek
    )

    data["day_of_month"] = (
        data[date_column].dt.day
    )

    data["day_of_year"] = (
        data[date_column].dt.dayofyear
    )

    data["week_of_year"] = (
        data[date_column]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Trend
    # --------------------------------------------------

    data["trend"] = np.arange(
        len(data),
        dtype=float,
    )

    # --------------------------------------------------
    # Cyclical weekly features
    # --------------------------------------------------

    data["sin_week"] = np.sin(
        2 * np.pi
        * data["day_of_week"]
        / 7
    )

    data["cos_week"] = np.cos(
        2 * np.pi
        * data["day_of_week"]
        / 7
    )

    # --------------------------------------------------
    # Cyclical annual features
    # --------------------------------------------------

    data["sin_year"] = np.sin(
        2 * np.pi
        * data["day_of_year"]
        / 365.25
    )

    data["cos_year"] = np.cos(
        2 * np.pi
        * data["day_of_year"]
        / 365.25
    )

    # --------------------------------------------------
    # Lag features
    # --------------------------------------------------

    for lag in lags:

        if lag <= 0:
            raise ValueError(
                "Lag values must be positive."
            )

        data[
            f"lag_{lag}"
        ] = data[target_column].shift(
            lag
        )

    # --------------------------------------------------
    # Rolling statistics
    #
    # IMPORTANT:
    # shift(1) happens BEFORE rolling.
    #
    # This guarantees that the rolling statistic
    # does not contain today's target value.
    # --------------------------------------------------

    shifted_target = data[
        target_column
    ].shift(1)

    for window in rolling_windows:

        if window <= 0:
            raise ValueError(
                "Rolling windows must be positive."
            )

        data[
            f"rolling_mean_{window}"
        ] = (
            shifted_target
            .rolling(window)
            .mean()
        )

        data[
            f"rolling_std_{window}"
        ] = (
            shifted_target
            .rolling(window)
            .std()
        )

        data[
            f"rolling_min_{window}"
        ] = (
            shifted_target
            .rolling(window)
            .min()
        )

        data[
            f"rolling_max_{window}"
        ] = (
            shifted_target
            .rolling(window)
            .max()
        )

    # --------------------------------------------------
    # Recent demand dynamics
    # --------------------------------------------------

    data["lag_ratio_1_7"] = (
        data["lag_1"]
        / (data["lag_7"] + 1e-6)
    )

    data["lag_ratio_7_30"] = (
        data["lag_7"]
        / (data["lag_30"] + 1e-6)
    )

    data["rolling_ratio_7_30"] = (
        data["rolling_mean_7"]
        / (
            data["rolling_mean_30"]
            + 1e-6
        )
    )

    # --------------------------------------------------
    # Target column remains separate.
    #
    # Rows containing insufficient lag history are
    # removed only AFTER all features are generated.
    # --------------------------------------------------

    data = data.dropna(
        subset=[
            target_column,
        ]
    )

    return data


def get_feature_columns(
    dataframe: pd.DataFrame,
    target_column: str,
    date_column: str = "datum",
) -> list[str]:
    """
    Return ML feature columns while excluding
    date and target columns.
    """

    excluded = {
        date_column,
        target_column,
    }

    return [
        column
        for column in dataframe.columns
        if column not in excluded
    ]


def build_supervised_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    date_column: str = "datum",
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X/y for supervised forecasting.

    The target is shifted into the future by `horizon`.

    Example:
        horizon=1
            features at t → demand at t+1

        horizon=7
            features at t → demand at t+7

        horizon=30
            features at t → demand at t+30
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    features = create_forecasting_features(
        dataframe=dataframe,
        target_column=target_column,
        date_column=date_column,
    )

    future_target = (
        features[target_column]
        .shift(-horizon)
    )

    feature_columns = get_feature_columns(
        features,
        target_column=target_column,
        date_column=date_column,
    )

    X = features[
        feature_columns
    ].copy()

    y = future_target.copy()

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid].reset_index(
        drop=True
    )

    y = y.loc[valid].reset_index(
        drop=True
    )

    return X, y
