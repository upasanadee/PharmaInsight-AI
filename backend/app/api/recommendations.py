from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECOMMENDATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "intelligence"
    / "recommendations.csv"
)


def load_recommendations() -> pd.DataFrame:
    if not RECOMMENDATION_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Recommendation report not found.",
        )

    return pd.read_csv(RECOMMENDATION_FILE)


@router.get("/recommendations")
def recommendations() -> list[dict]:
    dataframe = load_recommendations()

    if dataframe.empty:
        return []

    return dataframe.to_dict(orient="records")


@router.get("/recommendations/{category}")
def category_recommendation(category: str) -> dict:
    dataframe = load_recommendations()

    match = dataframe[
        dataframe["category"].str.upper() == category.upper()
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found.",
        )

    return match.iloc[0].to_dict()
