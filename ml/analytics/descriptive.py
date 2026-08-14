from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_TARGET_COLUMNS: list[str] = [
    "M01AB",
    "M01AE",
    "N02BA",
    "N02BE",
    "N05B",
    "N05C",
    "R03",
    "R06",
]


def calculate_descriptive_statistics(
    df: pd.DataFrame,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each demand series.

    Returns one row per pharmaceutical category.
    """

    records = []

    for column in target_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        mean = series.mean()
        std = series.std()

        coefficient_of_variation = (
            std / mean
            if mean != 0
            else np.nan
        )

        records.append(
            {
                "category": column,
                "count": int(series.count()),
                "mean": mean,
                "median": series.median(),
                "std": std,
                "min": series.min(),
                "max": series.max(),
                "skewness": series.skew(),
                "kurtosis": series.kurt(),
                "coefficient_of_variation": (
                    coefficient_of_variation
                ),
                "zero_demand_pct": (
                    (series == 0).mean() * 100
                ),
            }
        )

    return pd.DataFrame(records)