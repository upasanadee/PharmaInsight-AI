"""
PharmaInsight AI — Final Tuned ML Forecast Benchmark

Final chronological evaluation of tuned XGBoost and LightGBM models.

Protocol
--------
1. Hyperparameters are selected using the validation set.
2. The test set is never used during tuning.
3. After selection, training + validation data are combined.
4. The selected model is retrained on training + validation.
5. Final performance is evaluated once on the untouched test set.

This produces the final ML forecasting results.
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

RANDOM_STATE = 42

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CONFIG_PATH = (
    OUTPUT_DIR / "ml_best_configurations.csv"
)


# ------------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Split supervised data chronologically.

    No random shuffling is performed.
    """

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    n = len(X)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    X_train = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    X_validation = X.iloc[
        train_end:validation_end
    ].copy()

    y_validation = y.iloc[
        train_end:validation_end
    ].copy()

    X_test = X.iloc[
        validation_end:
    ].copy()

    y_test = y.iloc[
        validation_end:
    ].copy()

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

def create_tuned_model(
    model_name: str,
    params: dict,
):
    """
    Create a model using the selected validation configuration.
    """

    if model_name == "XGBoost":

        return create_xgboost_model(
            random_state=RANDOM_STATE,
            **params,
        )

    if model_name == "LightGBM":

        return create_lightgbm_model(
            random_state=RANDOM_STATE,
            **params,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print(
        "PHARMAINSIGHT AI — FINAL TUNED ML BENCHMARK"
    )
    print("=" * 70)

    # --------------------------------------------------------------
    # Load selected configurations
    # --------------------------------------------------------------

    if not BEST_CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{BEST_CONFIG_PATH}\n\n"
            "Run tune_ml.py first."
        )

    best_configs = pd.read_csv(
        BEST_CONFIG_PATH
    )

    print()
    print(
        f"Loaded {len(best_configs)} "
        "best validation configurations."
    )

    # --------------------------------------------------------------
    # Load dataset
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

    all_results = []

    # --------------------------------------------------------------
    # Evaluate every category/model
    # --------------------------------------------------------------

    for category in TARGET_COLUMNS:

        print()
        print("=" * 70)
        print(
            f"Category: {category}"
        )
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

        # ----------------------------------------------------------
        # Combine training + validation
        # ----------------------------------------------------------

        X_train_final = pd.concat(
            [
                X_train,
                X_validation,
            ],
            axis=0,
        ).reset_index(
            drop=True
        )

        y_train_final = pd.concat(
            [
                y_train,
                y_validation,
            ],
            axis=0,
        ).reset_index(
            drop=True
        )

        print(
            f"Training + validation: "
            f"{len(X_train_final):,}"
        )

        print(
            f"Untouched test set: "
            f"{len(X_test):,}"
        )

        # ----------------------------------------------------------
        # Models
        # ----------------------------------------------------------

        category_configs = (
            best_configs[
                best_configs["category"]
                == category
            ]
        )

        for _, config_row in (
            category_configs.iterrows()
        ):

            model_name = config_row[
                "model"
            ]

            config_id = int(
                config_row["config_id"]
            )

            print()
            print(
                f"Evaluating {model_name} "
                f"(selected config {config_id})..."
            )

            # ------------------------------------------------------
            # Recover configuration
            #
            # The tuning CSV contains the selected
            # configuration parameters.
            # ------------------------------------------------------

            if model_name == "XGBoost":

                params = {
                    "n_estimators": int(
                        config_row[
                            "n_estimators"
                        ]
                    ),
                    "learning_rate": float(
                        config_row[
                            "learning_rate"
                        ]
                    ),
                    "max_depth": int(
                        config_row[
                            "max_depth"
                        ]
                    ),
                }

            elif model_name == "LightGBM":

                params = {
                    "n_estimators": int(
                        config_row[
                            "n_estimators"
                        ]
                    ),
                    "learning_rate": float(
                        config_row[
                            "learning_rate"
                        ]
                    ),
                    "num_leaves": int(
                        config_row[
                            "num_leaves"
                        ]
                    ),
                }

            else:

                raise ValueError(
                    f"Unsupported model: "
                    f"{model_name}"
                )

            print(
                "Selected parameters:"
            )

            for key, value in params.items():

                print(
                    f"  {key}: {value}"
                )

            # ------------------------------------------------------
            # Create and train final model
            # ------------------------------------------------------

            model = create_tuned_model(
                model_name=model_name,
                params=params,
            )

            model.fit(
                X_train_final,
                y_train_final,
            )

            # ------------------------------------------------------
            # Predict untouched test set
            # ------------------------------------------------------

            predictions = model.predict(
                X_test
            )

            predictions = np.asarray(
                predictions,
                dtype=float,
            )

            # Demand cannot be negative.
            predictions = np.maximum(
                predictions,
                0.0,
            )

            # ------------------------------------------------------
            # Evaluate
            # ------------------------------------------------------

            training_series = y_train_final

            metrics = evaluate_forecast(
                actual=y_test,
                predicted=predictions,
                training_series=training_series,
                seasonality=1,
            )

            print()
            print(
                "Final test metrics:"
            )

            for metric, value in metrics.items():

                print(
                    f"  {metric}: "
                    f"{value:.4f}"
                )

            all_results.append(
                {
                    "category": category,
                    "model": model_name,
                    "config_id": config_id,
                    **params,
                    **metrics,
                }
            )

    # --------------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------------
    # Rankings within category
    # --------------------------------------------------------------

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
    )

    # --------------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "final_tuned_ml_benchmark.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Overall model summary
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
        .sort_values(
            "MASE",
            ascending=True,
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "final_tuned_ml_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    # --------------------------------------------------------------
    # Display
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL TUNED ML TEST RESULTS"
    )
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
        "MASE_rank",
        "sMAPE_rank",
        "MAE_rank",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print(
        "AVERAGE FINAL TEST PERFORMANCE"
    )
    print("=" * 70)

    print(
        summary.to_string(
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # --------------------------------------------------------------
    # Best overall model
    # --------------------------------------------------------------

    best_model = summary.index[0]

    print()
    print(
        f"Best overall tuned model "
        f"by mean MASE: {best_model}"
    )

    print()
    print("Saved:")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()

