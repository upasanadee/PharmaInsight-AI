"""
PharmaInsight AI — Walk-Forward Baseline Benchmark

Evaluates:
    Naive
    Seasonal Naive
    Moving Average

across all pharmaceutical demand categories using
rolling-origin evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.loader import load_dataset

from ml.forecasting.walk_forward import (
    walk_forward_baseline_evaluation,
    summarize_walk_forward_results,
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

MODELS = [
    "Naive",
    "Seasonal Naive",
    "Moving Average",
]


def run_benchmark(
    dataframe: pd.DataFrame,
    target_columns: list[str],
    initial_train_size: int = 1000,
    horizon: int = 30,
    step_size: int = 30,
    seasonal_period: int = 7,
) -> pd.DataFrame:

    all_results = []

    for category in target_columns:

        print(
            f"\nEvaluating {category}..."
        )

        series = dataframe[
            category
        ].astype(float)

        for model in MODELS:

            result = (
                walk_forward_baseline_evaluation(
                    series=series,
                    model_name=model,
                    initial_train_size=initial_train_size,
                    horizon=horizon,
                    step_size=step_size,
                    seasonal_period=seasonal_period,
                )
            )

            result.insert(
                0,
                "category",
                category,
            )

            all_results.append(
                result
            )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    metric_columns = [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    ]

    summary = (
        results
        .groupby(
            [
                "category",
                "model",
            ]
        )[metric_columns]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    return summary


def create_ranking(
    results: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        results
        .groupby(
            [
                "category",
                "model",
            ]
        )[
            [
                "MAE",
                "RMSE",
                "sMAPE",
                "WAPE",
                "MASE",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary["MASE_rank"] = (
        summary
        .groupby("category")["MASE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    summary["sMAPE_rank"] = (
        summary
        .groupby("category")["sMAPE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    summary["MAE_rank"] = (
        summary
        .groupby("category")["MAE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    return summary.sort_values(
        [
            "category",
            "MASE_rank",
        ]
    ).reset_index(drop=True)


if __name__ == "__main__":

    print("=" * 75)
    print(
        "PHARMAINSIGHT AI — "
        "WALK-FORWARD BASELINE BENCHMARK"
    )
    print("=" * 75)

    daily = load_dataset("daily")

    results = run_benchmark(
        dataframe=daily,
        target_columns=TARGET_COLUMNS,
        initial_train_size=1000,
        horizon=30,
        step_size=30,
        seasonal_period=7,
    )

    summary = create_summary(
        results
    )

    ranking = create_ranking(
        results
    )

    print("\n")
    print("=" * 75)
    print("AVERAGE PERFORMANCE")
    print("=" * 75)

    print(
        ranking.to_string(
            index=False
        )
    )

    output_dir = Path(
        "reports"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_dir
        / "walk_forward_baseline_folds.csv",
        index=False,
    )

    ranking.to_csv(
        output_dir
        / "walk_forward_baseline_ranking.csv",
        index=False,
    )

    print("\n")
    print(
        "Saved:"
    )

    print(
        output_dir
        / "walk_forward_baseline_folds.csv"
    )

    print(
        output_dir
        / "walk_forward_baseline_ranking.csv"
    )