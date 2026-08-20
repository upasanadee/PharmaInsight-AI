"""
PharmaInsight AI — Forecast Alerts API
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.dashboard_service import (
    dashboard_service,
)


router = APIRouter()


@router.get("/alerts")
def alerts() -> list[dict]:

    df = dashboard_service.get_alerts()

    if df.empty:
        return []

    return df.to_dict(
        orient="records"
    )
