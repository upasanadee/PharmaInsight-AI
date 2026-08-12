from ml.preprocessing.loader import (
    DATASETS,
    load_dataset,
)
from ml.preprocessing.validator import (
    TARGET_COLUMNS,
    validate_dataset,
)


def audit_dataset(name: str) -> None:
    """
    Generate and print a data-quality audit
    for one dataset.
    """

    df = load_dataset(name)

    print("=" * 70)
    print(f"DATASET: {name.upper()}")
    print("=" * 70)

    print(f"Rows:        {len(df):,}")
    print(f"Columns:     {len(df.columns)}")

    print(
        "Date range:  "
        f"{df['datum'].min()} → "
        f"{df['datum'].max()}"
    )

    print(
        f"Duplicates:  {df.duplicated().sum():,}"
    )

    print(
        f"Missing:     {df.isna().sum().sum():,}"
    )

    print(
        f"Unique dates: "
        f"{df['datum'].nunique():,}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(
            f"  - {column}: "
            f"{df[column].dtype}"
        )

    print("\nRunning validation...")

    validate_dataset(
        df=df,
        frequency=name,
        target_columns=TARGET_COLUMNS,
    )

    print("Validation: PASS")
    print()


def main() -> None:
    """
    Audit every available dataset.
    """

    print("\nPHARMAINSIGHT AI — DATA AUDIT")
    print("=" * 70)
    print()

    for dataset_name in DATASETS:
        audit_dataset(dataset_name)


if __name__ == "__main__":
    main()