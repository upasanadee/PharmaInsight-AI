"""
PharmaInsight AI — Dashboard Data Service

Reads validated forecasting outputs and exposes them
to the API layer.

This service does not train models or modify forecasts.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from backend.app.core.config import (
    CATEGORY_FILE,
    DAILY_FILE,
    SUMMARY_FILE,
)


class DashboardDataService:
    """Access validated dashboard forecasting data."""

    def __init__(self) -> None:
        self._validate_files()

    @staticmethod
    def _validate_files() -> None:
        files = {
            "category summary": CATEGORY_FILE,
            "daily forecast": DAILY_FILE,
            "dashboard summary": SUMMARY_FILE,
        }

        missing = [
            f"{name}: {path}"
            for name, path in files.items()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Required dashboard data files are missing:\n"
                + "\n".join(missing)
            )

    @lru_cache(maxsize=1)
    def category_summary(self) -> pd.DataFrame:
        """Return category-level forecast summary."""

        df = pd.read_csv(CATEGORY_FILE)

        return df.copy()

    @lru_cache(maxsize=1)
    def daily_forecasts(self) -> pd.DataFrame:
        """Return daily production forecasts."""

        df = pd.read_csv(DAILY_FILE)

        df["datum"] = pd.to_datetime(
            df["datum"]
        )

        return df.sort_values(
            ["category", "datum"]
        ).copy()

    @lru_cache(maxsize=1)
    def dashboard_summary(self) -> pd.DataFrame:
        """Return global dashboard KPIs."""

        return pd.read_csv(
            SUMMARY_FILE
        ).copy()

    def get_category(
        self,
        category: str,
    ) -> dict | None:
        """Return one category summary."""

        df = self.category_summary()

        matches = df[
            df["category"].astype(str).str.upper()
            == category.upper()
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    def get_forecast(
        self,
        category: str,
    ) -> pd.DataFrame:
        """Return forecast rows for one category."""

        df = self.daily_forecasts()

        result = df[
            df["category"].astype(str).str.upper()
            == category.upper()
        ].copy()

        return result

    def get_alerts(self) -> pd.DataFrame:
        """Return categories requiring attention."""

        df = self.category_summary()

        return df[
            df["status"].astype(str).str.upper()
            != "OK"
        ].copy()


dashboard_service = DashboardDataService()
