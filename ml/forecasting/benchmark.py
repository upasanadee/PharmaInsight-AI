"""
PharmaInsight AI — Baseline Forecasting Benchmark

Runs chronological train/test evaluation for the baseline
forecasting models across all pharmaceutical demand categories.

Models:
    - Naive
    - Seasonal Naive
    - Moving Average

Metrics:
    - MAE
    - RMSE
    - sMAPE
    - WAPE
    - MASE
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.forecasting.baselines import (
    generate_baseline_forecasts,
)

from ml.forecasting.metrics import (
    evaluate_forecast,
)

from ml.forecasting.splitter import (
    chronological_split,
)


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


def run_baseline_benchmark(
    dataframe: pd.DataFrame,
    target_columns: list[str],
    date_column: str = "datum",
    seasonal_period: int = 7,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Run baseline forecasting benchmark.

    The validation period is used to assess model behaviour,
    while the final test period remains chronologically later.

    Parameters
    ----------
    dataframe:
        Daily pharmaceutical demand dataframe.

    target_columns:
        Demand category columns.

    date_column:
        Timestamp column.

    seasonal_period:
        Seasonal period used by Seasonal Naive and MASE.
        For daily data, 7 represents weekly seasonality.

    train_ratio:
        Fraction used for training.

    validation_ratio:
        Fraction used for validation.

    test_ratio:
        Fraction used for testing.

    Returns
    -------
    pandas.DataFrame
        Benchmark results.
    """

    train, validation, test = chronological_split(
        dataframe=dataframe,
        date_column=date_column,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    print("=" * 70)
    print("PHARMAINSIGHT AI — BASELINE FORECAST BENCHMARK")
    print("=" * 70)

    print(f"Total observations: {len(dataframe):,}")
    print(f"Training observations: {len(train):,}")
    print(f"Validation observations: {len(validation):,}")
    print(f"Test observations: {len(test):,}")

    print(
        f"\nTraining period: "
        f"{train[date_column].min()} → "
        f"{train[date_column].max()}"
    )

    print(
        f"Validation period: "
        f"{validation[date_column].min()} → "
        f"{validation[date_column].max()}"
    )

    print(
        f"Test period: "
        f"{test[date_column].min()} → "
        f"{test[date_column].max()}"
    )

    records = []

    # ----------------------------------------------------------
    # Benchmark on validation set
    # ----------------------------------------------------------

    for category in target_columns:

        training_series = train[category].astype(float)

        actual = validation[
            category
        ].astype(float).to_numpy()

        forecasts = generate_baseline_forecasts(
            training_series=training_series,
            horizon=len(validation),
            seasonal_period=seasonal_period,
        )

        for model_name, prediction in forecasts.items():

            metrics = evaluate_forecast(
                actual=actual,
                predicted=prediction,
                training_series=training_series,
                seasonality=seasonal_period,
            )

            records.append(
                {
                    "category": category,
                    "model": model_name,
                    "evaluation_period": "validation",
                    **metrics,
                }
            )

    results = pd.DataFrame(records)

    # ----------------------------------------------------------
    # Ranking
    # ----------------------------------------------------------

    results["rank_by_sMAPE"] = (
        results
        .groupby("category")["sMAPE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    results["rank_by_MAE"] = (
        results
        .groupby("category")["MAE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    return results


def save_benchmark_results(
    results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save benchmark results to CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )


if __name__ == "__main__":

    from ml.preprocessing.loader import load_dataset

    daily = load_dataset("daily")

    results = run_baseline_benchmark(
        dataframe=daily,
        target_columns=TARGET_COLUMNS,
        date_column="datum",
        seasonal_period=7,
    )

    print("\n")
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(
        results[
            [
                "category",
                "model",
                "MAE",
                "RMSE",
                "sMAPE",
                "WAPE",
                "MASE",
                "rank_by_sMAPE",
            ]
        ].to_string(index=False)
    )

    save_benchmark_results(
        results,
        "reports/baseline_benchmark.csv",
    )