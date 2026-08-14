"""
PharmaInsight AI — Machine Learning Forecast Benchmark

Chronological benchmark for XGBoost and LightGBM.

Important:
- No random shuffling.
- Features are generated using historical observations only.
- Models are trained chronologically.
- The test period is strictly after the training/validation periods.
- Predictions are clipped at zero because pharmaceutical demand
  cannot be negative.
- MASE is calculated using the training series as the naive
  scaling reference.
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
from ml.forecasting.metrics import evaluate_forecast


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

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
):
    """
    Chronological train/validation/test split.

    No shuffling is performed.

    Returns
    -------
    X_train, y_train
    X_validation, y_validation
    X_test, y_test
    """

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 <= validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    n = len(X)

    train_end = int(n * train_ratio)
    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_validation = X.iloc[
        train_end:validation_end
    ].copy()

    y_validation = y.iloc[
        train_end:validation_end
    ].copy()

    X_test = X.iloc[validation_end:].copy()
    y_test = y.iloc[validation_end:].copy()

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ------------------------------------------------------------------
# Model training and evaluation
# ------------------------------------------------------------------

def run_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Train a model and evaluate its test predictions.

    MASE is scaled using the training series so that the test
    observations do not influence the scaling denominator.
    """

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    # Pharmaceutical demand cannot be negative.
    predictions = np.maximum(
        predictions,
        0.0,
    )

    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------

    metrics = evaluate_forecast(
        actual=y_test,
        predicted=predictions,
        training_series=y_train,
        seasonality=1,
    )

    return metrics, predictions


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("PHARMAINSIGHT AI — ML FORECAST BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load daily data
    # --------------------------------------------------------------

    dataframe = load_dataset(
        "daily"
    )

    print()
    print(
        f"Total daily observations: "
        f"{len(dataframe):,}"
    )

    print(
        f"Date range: "
        f"{dataframe['datum'].min()} "
        f"→ "
        f"{dataframe['datum'].max()}"
    )

    # --------------------------------------------------------------
    # Validate target columns
    # --------------------------------------------------------------

    missing_columns = [
        column
        for column in TARGET_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing target columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------------
    # Store results
    # --------------------------------------------------------------

    all_results = []

    # --------------------------------------------------------------
    # Evaluate every pharmaceutical category
    # --------------------------------------------------------------

    for category in TARGET_COLUMNS:

        print()
        print("=" * 70)
        print(f"Category: {category}")
        print("=" * 70)

        # ----------------------------------------------------------
        # Build leakage-safe supervised dataset
        # ----------------------------------------------------------

        X, y = build_supervised_dataset(
            dataframe=dataframe,
            target_column=category,
            horizon=1,
        )

        print(
            f"Supervised samples: "
            f"{len(X):,}"
        )

        print(
            f"Features: "
            f"{X.shape[1]}"
        )

        # ----------------------------------------------------------
        # Chronological split
        # ----------------------------------------------------------

        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        ) = chronological_split(
            X,
            y,
        )

        print()
        print(
            f"Training samples:   "
            f"{len(X_train):,}"
        )

        print(
            f"Validation samples: "
            f"{len(X_validation):,}"
        )

        print(
            f"Test samples:       "
            f"{len(X_test):,}"
        )

        # ----------------------------------------------------------
        # XGBoost
        # ----------------------------------------------------------

        print()
        print("Evaluating XGBoost...")

        xgb_model = create_xgboost_model(
            random_state=RANDOM_STATE,
        )

        xgb_metrics, _ = run_model(
            model=xgb_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        all_results.append(
            {
                "category": category,
                "model": "XGBoost",
                **xgb_metrics,
            }
        )

        print(xgb_metrics)

        # ----------------------------------------------------------
        # LightGBM
        # ----------------------------------------------------------

        print()
        print("Evaluating LightGBM...")

        lgb_model = create_lightgbm_model(
            random_state=RANDOM_STATE,
        )

        lgb_metrics, _ = run_model(
            model=lgb_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        all_results.append(
            {
                "category": category,
                "model": "LightGBM",
                **lgb_metrics,
            }
        )

        print(lgb_metrics)

    # ------------------------------------------------------------------
    # Create results dataframe
    # ------------------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    # ------------------------------------------------------------------
    # Rank models within each category
    # ------------------------------------------------------------------

    results["MASE_rank"] = (
        results
        .groupby("category")["MASE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    results["sMAPE_rank"] = (
        results
        .groupby("category")["sMAPE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    results["MAE_rank"] = (
        results
        .groupby("category")["MAE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    # ------------------------------------------------------------------
    # Overall ranking score
    # ------------------------------------------------------------------
    #
    # Lower is better.
    #
    # MASE, sMAPE and MAE each contribute equally.
    # ------------------------------------------------------------------

    results["overall_rank_score"] = (
        results["MASE_rank"]
        + results["sMAPE_rank"]
        + results["MAE_rank"]
    )

    results = results.sort_values(
        [
            "category",
            "overall_rank_score",
            "MASE",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save detailed benchmark
    # ------------------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "ml_forecast_benchmark.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary across categories
    # ------------------------------------------------------------------

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
        .sort_values(
            "MASE",
            ascending=True,
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "ml_forecast_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    # ------------------------------------------------------------------
    # Display final rankings
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ML RANKINGS")
    print("=" * 70)

    display_columns = [
        "category",
        "model",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
        "MASE_rank",
        "sMAPE_rank",
        "MAE_rank",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Average performance
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AVERAGE PERFORMANCE "
        "ACROSS CATEGORIES"
    )
    print("=" * 70)

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Overall winner
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL MODEL COMPARISON")
    print("=" * 70)

    overall = (
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
        .sort_values(
            "MASE",
            ascending=True,
        )
    )

    print(
        overall.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        "Best overall model by mean MASE: "
        f"{overall.index[0]}"
    )

    # ------------------------------------------------------------------
    # Output locations
    # ------------------------------------------------------------------

    print()
    print("Saved:")
    print(results_path)
    print(summary_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()