import numpy as np
import pandas as pd
import pytest

from ml.forecasting.features import create_forecasting_features
from ml.forecasting.metrics import mae, mase, mape, rmse, smape, wape
from ml.forecasting.splitter import chronological_split


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_mae():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    assert mae(actual, predicted) == pytest.approx(7 / 3)


def test_rmse():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    expected = np.sqrt((4 + 4 + 9) / 3)

    assert rmse(actual, predicted) == pytest.approx(expected)


def test_mape_excludes_zero_actuals():
    actual = np.array([0, 100, 200])
    predicted = np.array([50, 110, 180])

    # Zero-demand observation must be excluded.
    expected = ((10 / 100) + (20 / 200)) / 2 * 100

    assert mape(actual, predicted) == pytest.approx(expected)


def test_smape():
    actual = np.array([100, 200])
    predicted = np.array([110, 180])

    expected = (
        (
            2 * abs(100 - 110) / (abs(100) + abs(110))
            + 2 * abs(200 - 180) / (abs(200) + abs(180))
        )
        / 2
        * 100
    )

    assert smape(actual, predicted) == pytest.approx(expected)


def test_wape():
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 180, 330])

    expected = (10 + 20 + 30) / (100 + 200 + 300) * 100

    assert wape(actual, predicted) == pytest.approx(expected)


def test_mase():
    training = np.array([10, 12, 14, 16, 18])
    actual = np.array([20, 22])
    predicted = np.array([19, 21])

    # Naive in-sample errors:
    # |12-10|, |14-12|, |16-14|, |18-16| = 2
    # Forecast MAE = 1
    # MASE = 1 / 2 = 0.5
    assert mase(
        actual,
        predicted,
        training,
        seasonality=1,
    ) == pytest.approx(0.5)


def test_metrics_reject_mismatched_shapes():
    actual = np.array([1, 2, 3])
    predicted = np.array([1, 2])

    with pytest.raises(ValueError):
        mae(actual, predicted)


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError):
        mae(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def test_forecasting_features_create_expected_columns():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": np.arange(40, dtype=float),
        }
    )

    result = create_forecasting_features(
        dataframe,
        target_column="demand",
        lags=[1, 7, 30],
        rolling_windows=[7, 30],
    )

    expected_columns = {
        "datum",
        "demand",
        "year",
        "month",
        "quarter",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "trend",
        "sin_week",
        "cos_week",
        "sin_year",
        "cos_year",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_min_7",
        "rolling_max_7",
        "lag_ratio_1_7",
    }

    assert expected_columns.issubset(result.columns)


def test_forecasting_features_sort_dates():
    dates = pd.to_datetime(
        [
            "2020-01-03",
            "2020-01-01",
            "2020-01-02",
        ]
    )

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": [30, 10, 20],
        }
    )

    result = create_forecasting_features(
        dataframe,
        target_column="demand",
        lags=[1, 7, 30],
        rolling_windows=[7, 30],
    )

    assert result["datum"].is_monotonic_increasing
    assert result["demand"].tolist() == [10, 20, 30]


def test_lag_feature_uses_previous_observation():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": list(range(10, 50)),
        }
    )

    result = create_forecasting_features(
        dataframe,
        target_column="demand",
        lags=[1, 7, 30],
        rolling_windows=[7, 30],
    )

    assert pd.isna(result.loc[0, "lag_1"])
    assert result.loc[1, "lag_1"] == 10
    assert result.loc[4, "lag_1"] == 13


def test_rolling_features_do_not_use_current_target():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": list(range(10, 50)),
        }
    )

    result = create_forecasting_features(
        dataframe,
        target_column="demand",
        lags=[1, 7, 30],
        rolling_windows=[7, 30],
    )

    # At index 2, the rolling window must use demands
    # [10, 20], not [20, 30].
    assert result.loc[7, "rolling_mean_7"] == pytest.approx(13.0)


def test_invalid_lag_raises_error():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": list(range(1, 41)),
        }
    )

    with pytest.raises(ValueError):
        create_forecasting_features(
            dataframe,
            target_column="demand",
            lags=[0],
            rolling_windows=[2],
        )


def test_invalid_rolling_window_raises_error():
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": list(range(1, 41)),
        }
    )

    with pytest.raises(ValueError):
        create_forecasting_features(
            dataframe,
            target_column="demand",
            lags=[1],
            rolling_windows=[0],
        )


def test_missing_feature_columns_raise_error():
    dataframe = pd.DataFrame(
        {
            "datum": pd.date_range(
                "2020-01-01",
                periods=40,
                freq="D",
            ),
            "demand": list(range(1, 41)),
        }
    )

    with pytest.raises(ValueError):
        create_forecasting_features(
            dataframe,
            target_column="missing_target",
            lags=[1],
            rolling_windows=[2],
        )


# ---------------------------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------------------------

def test_chronological_split_preserves_time_order():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates,
            "demand": np.arange(100),
        }
    )

    train, validation, test = chronological_split(
        dataframe,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15

    assert train["datum"].max() < validation["datum"].min()
    assert validation["datum"].max() < test["datum"].min()

    assert train["datum"].is_monotonic_increasing
    assert validation["datum"].is_monotonic_increasing
    assert test["datum"].is_monotonic_increasing


def test_chronological_split_sorts_unsorted_input():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    dataframe = pd.DataFrame(
        {
            "datum": dates[::-1],
            "demand": np.arange(10),
        }
    )

    train, validation, test = chronological_split(
        dataframe,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    assert train["datum"].is_monotonic_increasing
    assert validation["datum"].is_monotonic_increasing
    assert test["datum"].is_monotonic_increasing


def test_invalid_split_ratios_raise_error():
    dataframe = pd.DataFrame(
        {
            "datum": pd.date_range(
                "2020-01-01",
                periods=10,
                freq="D",
            ),
            "demand": np.arange(10),
        }
    )

    with pytest.raises(ValueError):
        chronological_split(
            dataframe,
            train_ratio=0.70,
            validation_ratio=0.20,
            test_ratio=0.20,
        )


def test_split_requires_at_least_three_observations():
    dataframe = pd.DataFrame(
        {
            "datum": pd.date_range(
                "2020-01-01",
                periods=2,
                freq="D",
            ),
            "demand": [1, 2],
        }
    )

    with pytest.raises(ValueError):
        chronological_split(dataframe)
