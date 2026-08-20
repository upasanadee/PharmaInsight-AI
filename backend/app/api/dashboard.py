"""
PharmaInsight AI — Dashboard API
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.dashboard import (
    CategorySummary,
    DashboardSummary,
    HealthResponse,
)
from backend.app.services.dashboard_service import (
    dashboard_service,
)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    return HealthResponse(
        status="healthy",
        service="pharmainsight-backend",
        version="1.0.0",
    )


@router.get(
    "/dashboard/summary",
    response_model=DashboardSummary,
)
def dashboard_summary() -> DashboardSummary:

    df = dashboard_service.dashboard_summary()

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Dashboard summary is empty.",
        )

    row = df.iloc[0]

    return DashboardSummary(
        total_categories=int(
            row["total_categories"]
        ),
        flagged_categories=int(
            row["flagged_categories"]
        ),
        forecast_horizon_days=int(
            row["forecast_horizon_days"]
        ),
        total_forecast_demand=float(
            row["total_forecast_demand"]
        ),
        total_recent_30d_demand=float(
            row["total_recent_30d_demand"]
        ),
        overall_change_pct=float(
            row["overall_change_pct"]
        ),
        best_mase_category=str(
            row["best_mase_category"]
        ),
        best_mase_model=str(
            row["best_mase_model"]
        ),
        model_counts=str(
            row["model_counts"]
        ),
    )


@router.get(
    "/categories",
    response_model=list[CategorySummary],
)
def categories() -> list[CategorySummary]:

    df = dashboard_service.category_summary()

    return [
        CategorySummary(
            category=str(row["category"]),
            model=str(row["model"]),
            recent_30d_mean=float(
                row["recent_30d_mean"]
            ),
            forecast_30d_mean=float(
                row["forecast_30d_mean"]
            ),
            forecast_change_pct=float(
                row["forecast_change_pct"]
            ),
            MASE=float(row["MASE"]),
            status=str(row["status"]),
        )
        for _, row in df.iterrows()
    ]


@router.get(
    "/categories/{category}",
    response_model=CategorySummary,
)
def category(
    category: str,
) -> CategorySummary:

    row = dashboard_service.get_category(
        category
    )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found.",
        )

    return CategorySummary(
        category=str(row["category"]),
        model=str(row["model"]),
        recent_30d_mean=float(
            row["recent_30d_mean"]
        ),
        forecast_30d_mean=float(
            row["forecast_30d_mean"]
        ),
        forecast_change_pct=float(
            row["forecast_change_pct"]
        ),
        MASE=float(row["MASE"]),
        status=str(row["status"]),
    )
