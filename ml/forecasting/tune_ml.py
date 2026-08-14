"""
PharmaInsight AI — ML Hyperparameter Tuning

Tunes XGBoost and LightGBM using the chronological validation set.

The test set is NOT used for hyperparameter selection.
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

RANDOM_STATE = 42

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def chronological_split(X, y):
    """Split supervised data chronologically."""

    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    n = len(X)

    train_end = int(n * TRAIN_RATIO)
    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    return (
        X.iloc[:train_end].copy(),
        y.iloc[:train_end].copy(),
        X.iloc[train_end:validation_end].copy(),
        y.iloc[train_end:validation_end].copy(),
        X.iloc[validation_end:].copy(),
        y.iloc[validation_end:].copy(),
    )


# ------------------------------------------------------------------
# Candidate configurations
# ------------------------------------------------------------------

XGBOOST_CONFIGS = [
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
]


LIGHTGBM_CONFIGS = [
    {
        "n_estimators": 300,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 700,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
]


def evaluate_configuration(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """Train on training data and evaluate on validation data."""

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    predictions = np.maximum(
        predictions,
        0.0,
    )

    return evaluate_forecast(
        actual=y_validation,
        predicted=predictions,
        training_series=y_train,
        seasonality=1,
    )


def main():

    dataframe = load_dataset("daily")

    print()
    print("=" * 70)
    print("PHARMAINSIGHT AI — ML HYPERPARAMETER TUNING")
    print("=" * 70)

    all_results = []

    for category in TARGET_COLUMNS:

        print()
        print("=" * 70)
        print(f"Category: {category}")
        print("=" * 70)

        X, y = build_supervised_dataset(
            dataframe=dataframe,
            target_column=category,
            horizon=1,
        )

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

        print(
            f"Training:   {len(X_train):,}"
        )
        print(
            f"Validation: {len(X_validation):,}"
        )
        print(
            f"Test:       {len(X_test):,}"
        )

        # ----------------------------------------------------------
        # XGBoost
        # ----------------------------------------------------------

        for config_id, config in enumerate(
            XGBOOST_CONFIGS,
            start=1,
        ):

            print(
                f"XGBoost configuration {config_id}/"
                f"{len(XGBOOST_CONFIGS)}"
            )

            model = create_xgboost_model(
                random_state=RANDOM_STATE,
                **config,
            )

            metrics = evaluate_configuration(
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
            )

            all_results.append(
                {
                    "category": category,
                    "model": "XGBoost",
                    "config_id": config_id,
                    **config,
                    **metrics,
                }
            )

        # ----------------------------------------------------------
        # LightGBM
        # ----------------------------------------------------------

        for config_id, config in enumerate(
            LIGHTGBM_CONFIGS,
            start=1,
        ):

            print(
                f"LightGBM configuration {config_id}/"
                f"{len(LIGHTGBM_CONFIGS)}"
            )

            model = create_lightgbm_model(
                random_state=RANDOM_STATE,
                **config,
            )

            metrics = evaluate_configuration(
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
            )

            all_results.append(
                {
                    "category": category,
                    "model": "LightGBM",
                    "config_id": config_id,
                    **config,
                    **metrics,
                }
            )

    results = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------------
    # Rank configurations within each category/model
    # --------------------------------------------------------------

    results["validation_rank"] = (
        results
        .groupby(
            ["category", "model"]
        )["MASE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    results = results.sort_values(
        [
            "category",
            "model",
            "MASE",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # Select best configuration per category/model
    # --------------------------------------------------------------

    best = (
        results
        .sort_values(
            [
                "category",
                "model",
                "MASE",
            ]
        )
        .groupby(
            ["category", "model"],
            as_index=False,
        )
        .first()
    )

    # --------------------------------------------------------------
    # Save all tuning results
    # --------------------------------------------------------------

    all_path = (
        OUTPUT_DIR
        / "ml_hyperparameter_tuning.csv"
    )

    results.to_csv(
        all_path,
        index=False,
    )

    best_path = (
        OUTPUT_DIR
        / "ml_best_configurations.csv"
    )

    best.to_csv(
        best_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST VALIDATION CONFIGURATIONS")
    print("=" * 70)

    display_columns = [
        "category",
        "model",
        "config_id",
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "MASE",
        "validation_rank",
    ]

    print(
        best[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved:")
    print(all_path)
    print(best_path)


if __name__ == "__main__":
    main()
