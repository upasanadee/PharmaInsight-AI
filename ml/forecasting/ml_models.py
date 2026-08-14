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


# ------------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------------

def create_xgboost_model(
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    min_child_weight: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBRegressor:
    """
    Create an XGBoost regression model.

    Default parameters reproduce the original baseline model.
    Optional parameters allow chronological hyperparameter tuning.
    """

    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=random_state,
        n_jobs=-1,
    )


# ------------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------------

def create_lightgbm_model(
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> LGBMRegressor:
    """
    Create a LightGBM regression model.

    Default parameters reproduce the original baseline model.
    Optional parameters allow chronological hyperparameter tuning.
    """

    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


# ------------------------------------------------------------------
# Feature preparation
# ------------------------------------------------------------------

def prepare_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the feature matrix before model fitting.

    Missing and infinite values are explicitly checked.
    Missing values are not silently imputed.
    """

    X = X.copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().any().any():

        missing = X.isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            "Feature matrix contains missing values:\n"
            f"{missing}"
        )

    return X


# ------------------------------------------------------------------
# Fit and predict
# ------------------------------------------------------------------

def fit_and_predict(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Fit a forecasting model and generate predictions.
    """

    X_train = prepare_features(
        X_train
    )

    X_test = prepare_features(
        X_test
    )

    y_train = pd.Series(
        y_train
    ).astype(float)

    if y_train.isna().any():
        raise ValueError(
            "Training target contains NaN values."
        )

    model.fit(
        X_train,
        y_train,
    )

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

    return predictions