from collections.abc import Iterable

import pandas as pd


DATE_COLUMN = "datum"

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


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate that all required columns exist.
    """

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def validate_no_missing_values(
    df: pd.DataFrame,
) -> None:
    """
    Validate that the dataset contains no missing values.
    """

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            "Missing values detected:\n"
            f"{missing}"
        )


def validate_no_duplicate_rows(
    df: pd.DataFrame,
) -> None:
    """
    Validate that the dataset contains no duplicate rows.
    """

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate rows."
        )


def validate_unique_dates(
    df: pd.DataFrame,
    date_column: str = DATE_COLUMN,
) -> None:
    """
    Validate that each observation has a unique timestamp.
    """

    duplicate_dates = int(
        df[date_column].duplicated().sum()
    )

    if duplicate_dates > 0:
        raise ValueError(
            f"Found {duplicate_dates:,} duplicate "
            f"timestamps in '{date_column}'."
        )


def validate_non_negative_targets(
    df: pd.DataFrame,
    target_columns: Iterable[str],
) -> None:
    """
    Validate that all sales values are non-negative.
    """

    negative_counts: dict[str, int] = {}

    for column in target_columns:
        if column not in df.columns:
            continue

        negative_count = int(
            (df[column] < 0).sum()
        )

        if negative_count > 0:
            negative_counts[column] = negative_count

    if negative_counts:
        raise ValueError(
            "Negative sales values detected: "
            f"{negative_counts}"
        )


def validate_datetime_column(
    df: pd.DataFrame,
    column: str = DATE_COLUMN,
) -> None:
    """
    Validate the dataset's datetime column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Required datetime column '{column}' "
            "is missing."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df[column]
    ):
        raise TypeError(
            f"Column '{column}' must contain "
            "datetime values."
        )

    if df[column].isna().any():
        raise ValueError(
            f"Column '{column}' contains invalid "
            "or missing datetime values."
        )

    if not df[column].is_monotonic_increasing:
        raise ValueError(
            f"Column '{column}' is not "
            "chronologically sorted."
        )


def validate_temporal_frequency(
    df: pd.DataFrame,
    frequency: str,
    date_column: str = DATE_COLUMN,
) -> None:
    """
    Validate the expected temporal frequency.

    Expected frequencies:

    hourly  -> 1 hour
    daily   -> 1 day
    weekly  -> 7 days
    monthly -> consecutive calendar months
    """

    dates = df[date_column].sort_values()

    if frequency == "hourly":
        expected_delta = pd.Timedelta(hours=1)

        deltas = dates.diff().dropna()

        invalid = deltas[deltas != expected_delta]

        if not invalid.empty:
            raise ValueError(
                f"Hourly dataset contains "
                f"{len(invalid):,} invalid time intervals."
            )

        return

    if frequency == "daily":
        expected_delta = pd.Timedelta(days=1)

        deltas = dates.diff().dropna()

        invalid = deltas[deltas != expected_delta]

        if not invalid.empty:
            raise ValueError(
                f"Daily dataset contains "
                f"{len(invalid):,} invalid time intervals."
            )

        return

    if frequency == "weekly":
        expected_delta = pd.Timedelta(days=7)

        deltas = dates.diff().dropna()

        invalid = deltas[deltas != expected_delta]

        if not invalid.empty:
            raise ValueError(
                f"Weekly dataset contains "
                f"{len(invalid):,} invalid time intervals."
            )

        return

    if frequency == "monthly":
        periods = (
            dates
            .dt.to_period("M")
            .reset_index(drop=True)
        )

        expected_periods = pd.period_range(
            start=periods.iloc[0],
            end=periods.iloc[-1],
            freq="M",
        )

        expected_series = pd.Series(
            expected_periods
        ).reset_index(drop=True)

        if not periods.equals(expected_series):
            raise ValueError(
                "Monthly dataset contains missing "
                "or unexpected calendar months."
            )

        return

    raise ValueError(
        f"Unsupported frequency '{frequency}'. "
        "Expected hourly, daily, weekly, or monthly."
    )


def validate_dataset(
    df: pd.DataFrame,
    frequency: str,
    target_columns: Iterable[str] = TARGET_COLUMNS,
) -> None:
    """
    Run all core data-quality checks for a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    frequency : str
        Dataset frequency.

    target_columns : Iterable[str]
        Expected sales columns.
    """

    required_columns = [
        DATE_COLUMN,
        *target_columns,
    ]

    validate_required_columns(
        df,
        required_columns,
    )

    validate_datetime_column(
        df,
        DATE_COLUMN,
    )

    validate_unique_dates(
        df,
        DATE_COLUMN,
    )

    validate_no_missing_values(df)

    validate_no_duplicate_rows(df)

    validate_non_negative_targets(
        df,
        target_columns,
    )

    validate_temporal_frequency(
        df,
        frequency,
        DATE_COLUMN,
    )