"""
PharmaInsight AI — Model Performance API
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "final_model_comparison.csv"
)


@router.get("/model-performance")
def model_performance() -> list[dict]:

    if not MODEL_FILE.exists():
        return []

    df = pd.read_csv(
        MODEL_FILE
    )

    return df.to_dict(
        orient="records"
    )
