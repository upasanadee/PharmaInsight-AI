from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "pharmainsight-backend"
    assert data["version"] == "1.0.0"


def test_dashboard_summary():
    response = client.get("/dashboard/summary")

    assert response.status_code == 200

    data = response.json()

    required_fields = {
        "total_categories",
        "flagged_categories",
        "forecast_horizon_days",
        "total_forecast_demand",
        "total_recent_30d_demand",
        "overall_change_pct",
        "best_mase_category",
        "best_mase_model",
        "model_counts",
    }

    assert required_fields.issubset(data.keys())


def test_categories():
    response = client.get("/categories")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)

    if data:
        required_fields = {
            "category",
            "model",
            "recent_30d_mean",
            "forecast_30d_mean",
            "forecast_change_pct",
            "MASE",
            "status",
        }

        assert required_fields.issubset(data[0].keys())


def test_unknown_category_returns_404():
    response = client.get("/categories/__definitely_not_a_real_category__")

    assert response.status_code == 404


def test_unknown_forecast_returns_404():
    response = client.get(
        "/forecasts/__definitely_not_a_real_category__"
    )

    assert response.status_code == 404


def test_alerts():
    response = client.get("/alerts")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)


def test_model_performance():
    response = client.get("/model-performance")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
