"""
PharmaInsight AI — Final Forecasting Benchmark

Combines statistical and tuned ML forecasting models.

Models:
    Statistical:
        Naive
        Seasonal Naive
        Moving Average
        ETS
        SARIMA

    Machine Learning:
        XGBoost
        LightGBM

Primary metric:
    MASE

All results come from chronological evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports")

STATISTICAL_FILE = (
    REPORT_DIR / "statistical_forecast_rankings.csv"
)

ML_FILE = (
    REPORT_DIR / "final_tuned_ml_benchmark.csv"
)

OUTPUT_FILE = (
    REPORT_DIR / "final_model_comparison.csv"
)

BEST_FILE = (
    REPORT_DIR / "best_model_by_category.csv"
)


def load_statistical_results() -> pd.DataFrame:
    """Load statistical forecasting results."""

    df = pd.read_csv(STATISTICAL_FILE)

    required = {
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns in statistical results: "
            f"{sorted(missing)}"
        )

    return df[
        [
            "category",
            "model",
            "MAE",
            "RMSE",
            "sMAPE",
            "WAPE",
            "MASE",
        ]
    ].copy()


def load_ml_results() -> pd.DataFrame:
    """Load final tuned ML test results."""

    df = pd.read_csv(ML_FILE)

    required = {
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns in ML results: "
            f"{sorted(missing)}"
        )

    return df[
        [
            "category",
            "model",
            "MAE",
            "RMSE",
            "sMAPE",
            "WAPE",
            "MASE",
        ]
    ].copy()


def main() -> None:

    print()
    print("=" * 78)
    print("PHARMAINSIGHT AI — FINAL FORECASTING BENCHMARK")
    print("=" * 78)

    statistical = load_statistical_results()
    ml = load_ml_results()

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------

    results = pd.concat(
        [
            statistical,
            ml,
        ],
        ignore_index=True,
    )

    # Ensure numeric metrics.
    metric_columns = [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
    ]

    for column in metric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    # --------------------------------------------------------------
    # Check for duplicates
    # --------------------------------------------------------------

    duplicates = results.duplicated(
        subset=["category", "model"],
        keep=False,
    )

    if duplicates.any():

        duplicate_rows = results.loc[
            duplicates,
            ["category", "model"],
        ]

        raise ValueError(
            "Duplicate category/model combinations found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    # --------------------------------------------------------------
    # Rank all models within each category
    # --------------------------------------------------------------

    results["MASE_rank"] = (
        results.groupby("category")["MASE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    results["MAE_rank"] = (
        results.groupby("category")["MAE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    results["sMAPE_rank"] = (
        results.groupby("category")["sMAPE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    # MASE is primary.
    results["overall_rank_score"] = (
        results["MASE_rank"]
        + results["MAE_rank"]
        + results["sMAPE_rank"]
    )

    results = results.sort_values(
        [
            "category",
            "MASE_rank",
            "MASE",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    # --------------------------------------------------------------
    # Best model per category
    # --------------------------------------------------------------

    best = (
        results
        .sort_values(
            [
                "category",
                "MASE",
                "MAE",
                "sMAPE",
            ],
            ascending=True,
        )
        .groupby(
            "category",
            as_index=False,
        )
        .first()
    )

    best = best[
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
    ]

    # --------------------------------------------------------------
    # Save complete comparison
    # --------------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    best.to_csv(
        BEST_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Display detailed results
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("ALL MODELS — FINAL TEST PERFORMANCE")
    print("=" * 78)

    display_columns = [
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
        "MASE_rank",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------------
    # Best model per category
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("BEST MODEL BY CATEGORY")
    print("=" * 78)

    print(
        best.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------------
    # Count wins
    # --------------------------------------------------------------

    wins = (
        best["model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="category_wins")
    )

    print()
    print("=" * 78)
    print("MODEL WIN COUNT")
    print("=" * 78)

    print(wins.to_string(index=False))

    # --------------------------------------------------------------
    # Overall average performance
    # --------------------------------------------------------------

    summary = (
        results
        .groupby("model")[
            [
                "MAE",
                "RMSE",
                "sMAPE",
                "WAPE",
                "MASE",
            ]
        ]
        .mean()
        .sort_values("MASE")
    )

    print()
    print("=" * 78)
    print("AVERAGE PERFORMANCE ACROSS ALL CATEGORIES")
    print("=" * 78)

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_overall = summary.index[0]

    print()
    print(
        f"Best overall model by mean MASE: "
        f"{best_overall}"
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print(BEST_FILE)


if __name__ == "__main__":
    main()
