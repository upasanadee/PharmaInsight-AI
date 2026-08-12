from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


DATASETS = {
    "hourly": "saleshourly.csv",
    "daily": "salesdaily.csv",
    "weekly": "salesweekly.csv",
    "monthly": "salesmonthly.csv",
}


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a PharmaInsight AI raw dataset.

    Parameters
    ----------
    name : str
        Dataset frequency:
        'hourly', 'daily', 'weekly', or 'monthly'.

    Returns
    -------
    pd.DataFrame
        Dataset with the `datum` column parsed as datetime.

    Raises
    ------
    ValueError
        If the dataset name is unsupported.
    FileNotFoundError
        If the dataset file does not exist.
    """

    if name not in DATASETS:
        valid_names = ", ".join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Expected one of: {valid_names}."
        )

    file_path = RAW_DATA_DIR / DATASETS[name]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if "datum" not in df.columns:
        raise ValueError(
            f"Dataset '{name}' does not contain the required "
            "'datum' column."
        )

    df["datum"] = pd.to_datetime(
        df["datum"],
        errors="raise",
    )

    return df


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Load all PharmaInsight AI datasets.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by dataset frequency.
    """

    return {
        name: load_dataset(name)
        for name in DATASETS
    }