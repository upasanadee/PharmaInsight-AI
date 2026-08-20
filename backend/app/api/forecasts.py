"""
PharmaInsight AI — Forecast API
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.dashboard import ForecastPoint
from backend.app.services.dashboard_service import (
    dashboard_service,
)


router = APIRouter()


@router.get(
    "/forecasts/{category}",
    response_model=list[ForecastPoint],
)
def forecast(
    category: str,
) -> list[ForecastPoint]:

    df = dashboard_service.get_forecast(
        category
    )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No forecast found for "
                f"category '{category}'."
            ),
        )

    return [
        ForecastPoint(
            datum=row["datum"].strftime(
                "%Y-%m-%d"
            ),
            category=str(row["category"]),
            model=str(row["model"]),
            forecast=float(row["forecast"]),
        )
        for _, row in df.iterrows()
    ]
