"""
PharmaInsight AI — Backend Configuration
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORTS_DIR = PROJECT_ROOT / "reports" / "dashboard"

CATEGORY_FILE = REPORTS_DIR / "category_summary.csv"
DAILY_FILE = REPORTS_DIR / "forecast_daily.csv"
SUMMARY_FILE = REPORTS_DIR / "dashboard_summary.csv"

API_PREFIX = "/api/v1"

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "PHARMAINSIGHT_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
