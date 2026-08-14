"""
PharmaInsight AI — Statistical Forecast Benchmark
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.loader import load_dataset

from ml.forecasting.walk_forward import (
    walk_forward_baseline_evaluation,
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
    "ETS",
    "SARIMA",
]


def run_benchmark():

    daily = load_dataset("daily")

    all_results = []

    for category in TARGET_COLUMNS:

        print()
        print("=" * 70)
        print(f"Category: {category}")
        print("=" * 70)

        for model in MODELS:

            print(
                f"  Evaluating {model}..."
            )

            result = (
                walk_forward_baseline_evaluation(
                    series=daily[category],
                    model_name=model,
                    initial_train_size=1000,
                    horizon=30,
                    step_size=30,
                    seasonal_period=7,
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


def create_summary(results):

    metrics = [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    ]

    summary = (
        results
        .groupby(
            ["category", "model"]
        )[metrics]
        .mean()
        .reset_index()
    )

    return summary


def create_rankings(summary):

    summary = summary.copy()

    for metric in [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    ]:

        summary[
            f"{metric}_rank"
        ] = (
            summary
            .groupby("category")[metric]
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
    )


if __name__ == "__main__":

    print(
        "\n"
        "PHARMAINSIGHT AI — "
        "STATISTICAL FORECAST BENCHMARK\n"
    )

    results = run_benchmark()

    summary = create_summary(
        results
    )

    rankings = create_rankings(
        summary
    )

    output_dir = Path(
        "reports"
    )

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    results.to_csv(
        output_dir
        / "statistical_forecast_folds.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "statistical_forecast_summary.csv",
        index=False,
    )

    rankings.to_csv(
        output_dir
        / "statistical_forecast_rankings.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL RANKINGS BY MASE")
    print("=" * 70)

    print(
        rankings[
            [
                "category",
                "model",
                "MAE",
                "RMSE",
                "sMAPE",
                "WAPE",
                "MASE",
                "MASE_rank",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(
        output_dir
        / "statistical_forecast_folds.csv"
    )
    print(
        output_dir
        / "statistical_forecast_summary.csv"
    )
    print(
        output_dir
        / "statistical_forecast_rankings.csv"
    )