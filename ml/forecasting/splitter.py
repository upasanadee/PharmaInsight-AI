"""
PharmaInsight AI — Chronological Forecasting Splitter

Time-series data must never be randomly shuffled before forecasting.
This module provides chronological train/validation/test splitting.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    dataframe: pd.DataFrame,
    date_column: str = "datum",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataframe chronologically.

    No random shuffling is performed.

    Parameters
    ----------
    dataframe:
        Input time-series dataframe.

    date_column:
        Name of timestamp column.

    train_ratio:
        Fraction assigned to training.

    validation_ratio:
        Fraction assigned to validation.

    test_ratio:
        Fraction assigned to testing.

    Returns
    -------
    train, validation, test
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if date_column not in dataframe.columns:
        raise ValueError(
            f"Date column '{date_column}' not found in dataframe."
        )

    if train_ratio <= 0:
        raise ValueError("train_ratio must be greater than 0.")

    if validation_ratio < 0:
        raise ValueError("validation_ratio cannot be negative.")

    if test_ratio <= 0:
        raise ValueError("test_ratio must be greater than 0.")

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio "
            "must equal 1.0."
        )

    data = dataframe.copy()

    # Ensure chronological ordering.
    data[date_column] = pd.to_datetime(
        data[date_column]
    )

    data = data.sort_values(
        date_column
    ).reset_index(drop=True)

    n = len(data)

    if n < 3:
        raise ValueError(
            "At least 3 observations are required."
        )

    train_end = int(n * train_ratio)

    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    # Make sure every split contains observations.
    train_end = max(1, train_end)

    if validation_ratio > 0:
        validation_end = max(
            train_end + 1,
            validation_end,
        )
    else:
        validation_end = train_end

    validation_end = min(
        validation_end,
        n - 1,
    )

    train = data.iloc[
        :train_end
    ].copy()

    validation = data.iloc[
        train_end:validation_end
    ].copy()

    test = data.iloc[
        validation_end:
    ].copy()

    if len(validation) == 0 and validation_ratio > 0:
        raise ValueError(
            "Validation split is empty. "
            "Increase dataset size or validation_ratio."
        )

    if len(test) == 0:
        raise ValueError(
            "Test split is empty. "
            "Increase dataset size or adjust split ratios."
        )

    return train, validation, test
