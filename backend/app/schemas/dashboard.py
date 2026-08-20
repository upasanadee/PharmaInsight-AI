"""
PharmaInsight AI — Dashboard API Schemas
"""

from __future__ import annotations

from pydantic import BaseModel


class DashboardSummary(BaseModel):
    total_categories: int
    flagged_categories: int
    forecast_horizon_days: int

    total_forecast_demand: float
    total_recent_30d_demand: float
    overall_change_pct: float

    best_mase_category: str
    best_mase_model: str
    model_counts: str


class CategorySummary(BaseModel):
    category: str
    model: str

    recent_30d_mean: float
    forecast_30d_mean: float
    forecast_change_pct: float

    MASE: float
    status: str


class ForecastPoint(BaseModel):
    datum: str
    category: str
    model: str
    forecast: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
