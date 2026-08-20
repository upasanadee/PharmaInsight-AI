"""
PharmaInsight AI — FastAPI Application
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.alerts import router as alerts_router
from backend.app.api.dashboard import router as dashboard_router
from backend.app.api.forecasts import router as forecasts_router
from backend.app.api.models import router as models_router
from backend.app.core.config import (
    ALLOWED_ORIGINS,
    API_PREFIX,
)


app = FastAPI(
    title="PharmaInsight AI API",
    description=(
        "Production API for pharmaceutical demand "
        "forecasting and business analytics."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


app.include_router(
    dashboard_router,
    prefix=API_PREFIX,
    tags=["Dashboard"],
)

app.include_router(
    forecasts_router,
    prefix=API_PREFIX,
    tags=["Forecasts"],
)

app.include_router(
    models_router,
    prefix=API_PREFIX,
    tags=["Models"],
)

app.include_router(
    alerts_router,
    prefix=API_PREFIX,
    tags=["Alerts"],
)


@app.get("/")
def root() -> dict[str, str]:

    return {
        "application": "PharmaInsight AI",
        "service": "forecasting-api",
        "version": "1.0.0",
        "status": "running",
    }
