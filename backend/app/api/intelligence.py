from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

INTELLIGENCE_FILE = (
    Path("reports")
    / "intelligence"
    / "forecast_explanations.csv"
)


def load_intelligence() -> pd.DataFrame:
    if not INTELLIGENCE_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Intelligence report not found.",
        )

    dataframe = pd.read_csv(INTELLIGENCE_FILE)

    if "risk_factors" in dataframe.columns:
        dataframe["risk_factors"] = dataframe[
            "risk_factors"
        ].apply(
            lambda value: (
                ast.literal_eval(value)
                if isinstance(value, str) and value.strip()
                else []
            )
        )

    return dataframe


@router.get("/intelligence")
def intelligence() -> list[dict]:
    dataframe = load_intelligence()

    if dataframe.empty:
        return []

    return dataframe.to_dict(orient="records")


@router.get("/intelligence/{category}")
def category_intelligence(category: str) -> dict:
    dataframe = load_intelligence()

    match = dataframe[
        dataframe["category"].str.upper()
        == category.upper()
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found.",
        )

    return match.iloc[0].to_dict()
