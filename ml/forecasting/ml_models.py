"""
PharmaInsight AI — Machine-Learning Forecasting Models

Leakage-safe ML forecasting using XGBoost and LightGBM.

The models are trained only on chronological training data.
No random shuffling is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def create_xgboost_model(
    random_state: int = 42,
) -> XGBRegressor:
    """Create the baseline XGBoost regressor."""

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=random_state,
        n_jobs=-1,
    )


def create_lightgbm_model(
    random_state: int = 42,
) -> LGBMRegressor:
    """Create the baseline LightGBM regressor."""

    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def prepare_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the feature matrix before model fitting.

    The function does not impute missing values silently.
    Missing and infinite values are replaced with NaN and then
    checked explicitly.
    """

    X = X.copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        missing = X.isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            "Feature matrix contains missing values:\n"
            f"{missing}"
        )

    return X


def fit_and_predict(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Fit a forecasting model and generate predictions.
    """

    X_train = prepare_features(X_train)
    X_test = prepare_features(X_test)

    y_train = pd.Series(y_train).astype(float)

    if y_train.isna().any():
        raise ValueError("Training target contains NaN values.")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions = np.asarray(predictions, dtype=float)

    # Pharmaceutical demand cannot be negative.
    predictions = np.maximum(predictions, 0.0)

    return predictions
