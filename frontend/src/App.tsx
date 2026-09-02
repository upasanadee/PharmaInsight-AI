import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  BarChart3,
  Bell,
  ChevronRight,
  LayoutDashboard,
  Moon,
  Package,
  RefreshCw,
  Sun,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

type Theme = "dark" | "light";

type Summary = {
  total_categories: number;
  flagged_categories: number;
  forecast_horizon_days: number;
  total_forecast_demand: number;
  total_recent_30d_demand: number;
  overall_change_pct: number;
  best_mase_category: string;
  best_mase_model: string;
  model_counts: string;
};

type Category = {
  category: string;
  model: string;
  recent_30d_mean: number;
  forecast_30d_mean: number;
  forecast_change_pct: number;
  MASE: number;
  status: string;
};

type ForecastPoint = {
  datum: string;
  category: string;
  model: string;
  forecast: number;
};

type ModelPerformance = {
  category: string;
  model: string;
  MAE: number;
  RMSE: number;
  sMAPE: number;
  WAPE: number;
  MASE: number;
  overall_rank_score: number;
};

type Alert = {
  category: string;
  model: string;
  MASE: number;
  recent_30d_mean: number;
  forecast_30d_mean: number;
  forecast_change_pct: number;
  status: string;
  business_interpretation: string;
  forecast_min: number;
  forecast_max: number;
};

const api = axios.create({
  baseURL: "/api/v1",
});

function formatNumber(value: number) {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: 1,
  });
}

function formatPercent(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
}

function StatusBadge({ status }: { status: string }) {
  const isOk = status === "OK";

  return (
    <span className={`status-badge ${isOk ? "ok" : "warning"}`}>
      {!isOk && <AlertTriangle size={13} />}
      {isOk ? "OK" : status.replaceAll("_", " ")}
    </span>
  );
}

function App() {
  const [page, setPage] = useState("dashboard");

  const [summary, setSummary] = useState<Summary | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const [models, setModels] = useState<ModelPerformance[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);

  const [selectedCategory, setSelectedCategory] = useState("");
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  /*
   * ---------------------------------------------------------
   * THEME
   * ---------------------------------------------------------
   */

  const [theme, setTheme] = useState<Theme>(() => {
    const savedTheme = localStorage.getItem("pharmainsight-theme");

    return savedTheme === "light" ? "light" : "dark";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("pharmainsight-theme", theme);
  }, [theme]);

  /*
   * ---------------------------------------------------------
   * API DATA
   * ---------------------------------------------------------
   */

  const loadData = useCallback(async () => {
    try {
      setRefreshing(true);

      const [
        summaryRes,
        categoriesRes,
        modelsRes,
        alertsRes,
      ] = await Promise.all([
        api.get<Summary>("/dashboard/summary"),
        api.get<Category[]>("/categories"),
        api.get<ModelPerformance[]>("/model-performance"),
        api.get<Alert[]>("/alerts"),
      ]);

      setSummary(summaryRes.data);
      setCategories(categoriesRes.data);
      setModels(modelsRes.data);
      setAlerts(alertsRes.data);

      if (!selectedCategory && categoriesRes.data.length > 0) {
        setSelectedCategory(categoriesRes.data[0].category);
      }
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedCategory]);

  const loadForecast = async (category: string) => {
    if (!category) {
      return;
    }

    try {
      const response = await api.get<ForecastPoint[]>(
        `/forecasts/${encodeURIComponent(category)}`
      );

      setForecast(response.data);
    } catch (error) {
      console.error("Failed to load forecast:", error);
      setForecast([]);
    }
  };

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (selectedCategory) {
      loadForecast(selectedCategory);
    }
  }, [selectedCategory]);

  /*
   * ---------------------------------------------------------
   * DERIVED DATA
   * ---------------------------------------------------------
   */

  const selectedCategoryData = useMemo(
    () =>
      categories.find(
        (item) => item.category === selectedCategory
      ),
    [categories, selectedCategory]
  );

  /*
   * Best model for every category.
   *
   * This is calculated from the actual model-performance API
   * response instead of hardcoding model selections.
   */
  const bestModels = useMemo(() => {
    return categories
      .map((category) => {
        const categoryModels = models.filter(
          (model) => model.category === category.category
        );

        if (categoryModels.length === 0) {
          return null;
        }

        return [...categoryModels].sort(
          (a, b) =>
            a.overall_rank_score - b.overall_rank_score
        )[0];
      })
      .filter(
        (model): model is ModelPerformance =>
          model !== null
      );
  }, [categories, models]);

  /*
   * Count the models actually selected for each category.
   *
   * Example:
   * SARIMA: 5
   * LightGBM: 3
   *
   * This replaces the old hardcoded 5 / 3 values.
   */
  const modelCounts = useMemo(() => {
    const counts: Record<string, number> = {};

    bestModels.forEach((model) => {
      counts[model.model] = (counts[model.model] ?? 0) + 1;
    });

    return counts;
  }, [bestModels]);

  /*
   * Actual best MASE across the returned model benchmark.
   */
  const bestOverallModel = useMemo(() => {
    if (models.length === 0) {
      return null;
    }

    return [...models].sort(
      (a, b) => a.MASE - b.MASE
    )[0];
  }, [models]);

  /*
   * Theme-aware chart colors.
   *
   * CSS variables are defined in App.css / index.css.
   */
  const chartColors = {
    line: "var(--chart-line, #7c5cff)",
    grid: "var(--chart-grid, rgba(128, 128, 128, 0.25))",
    axis: "var(--chart-axis, #8b8b98)",
    tooltipBackground:
      "var(--tooltip-background, #15151f)",
    tooltipBorder:
      "var(--tooltip-border, rgba(255,255,255,0.12))",
    tooltipText:
      "var(--tooltip-text, #ffffff)",
  };

  /*
   * ---------------------------------------------------------
   * LOADING STATE
   * ---------------------------------------------------------
   */

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-logo">P</div>
        <h2>PharmaInsight AI</h2>
        <p>Loading forecasting intelligence...</p>
      </div>
    );
  }

  /*
   * ---------------------------------------------------------
   * MAIN APPLICATION
   * ---------------------------------------------------------
   */

  return (
    <div className="app-shell">
      {/* SIDEBAR */}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">P</div>

          <div>
            <strong>PharmaInsight</strong>
            <span>AI Intelligence Platform</span>
          </div>
        </div>

        <nav className="navigation">
          <button
            className={
              page === "dashboard" ? "active" : ""
            }
            onClick={() => setPage("dashboard")}
          >
            <LayoutDashboard size={18} />
            Dashboard
          </button>

          <button
            className={
              page === "forecasts" ? "active" : ""
            }
            onClick={() => setPage("forecasts")}
          >
            <BarChart3 size={18} />
            Forecasts
          </button>

          <button
            className={
              page === "categories" ? "active" : ""
            }
            onClick={() => setPage("categories")}
          >
            <Package size={18} />
            Categories
          </button>

          <button
            className={
              page === "models" ? "active" : ""
            }
            onClick={() => setPage("models")}
          >
            <TrendingUp size={18} />
            Models
          </button>

          <button
            className={
              page === "alerts" ? "active" : ""
            }
            onClick={() => setPage("alerts")}
          >
            <Bell size={18} />
            Alerts

            {alerts.length > 0 && (
              <span className="nav-count">
                {alerts.length}
              </span>
            )}
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="system-status">
            <span className="pulse" />
            API Operational
          </div>

          <span>PharmaInsight AI v1.0</span>
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">
        {/* TOP BAR */}

        <header className="topbar">
          <div>
            <span className="eyebrow">
              PHARMACEUTICAL SALES INTELLIGENCE
            </span>

            <h1>
              {page === "dashboard" &&
                "Executive Dashboard"}

              {page === "forecasts" &&
                "Demand Forecasts"}

              {page === "categories" &&
                "Category Performance"}

              {page === "models" &&
                "Model Performance"}

              {page === "alerts" &&
                "Forecast Alerts"}
            </h1>
          </div>

          <div className="topbar-actions">
            {/* THEME TOGGLE */}

            <button
              className="theme-toggle"
              onClick={() =>
                setTheme(
                  theme === "dark" ? "light" : "dark"
                )
              }
              title={
                theme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
              aria-label={
                theme === "dark"
                  ? "Switch to light mode"
                  : "Switch to dark mode"
              }
            >
              {theme === "dark" ? (
                <>
                  <Sun size={16} />
                  Light
                </>
              ) : (
                <>
                  <Moon size={16} />
                  Dark
                </>
              )}
            </button>

            {/* REFRESH */}

            <button
              className="refresh-button"
              onClick={loadData}
              disabled={refreshing}
            >
              <RefreshCw
                size={16}
                className={
                  refreshing ? "spin" : ""
                }
              />
              Refresh
            </button>
          </div>
        </header>

        {/* =================================================
            DASHBOARD
           ================================================= */}

        {page === "dashboard" && summary && (
          <>
            {/* HERO */}

            <section className="hero">
              <div>
                <span className="hero-label">
                  FORECAST OUTLOOK
                </span>

                <h2>
                  Demand expected to{" "}
                  <span
                    className={
                      summary.overall_change_pct >= 0
                        ? "positive"
                        : "negative"
                    }
                  >
                    {Math.abs(
                      summary.overall_change_pct
                    ).toFixed(1)}
                    %
                    {summary.overall_change_pct >= 0
                      ? " higher"
                      : " lower"}
                  </span>
                </h2>

                <p>
                  Based on a{" "}
                  {summary.forecast_horizon_days}
                  -day forecasting horizon across{" "}
                  {summary.total_categories}{" "}
                  pharmaceutical categories.
                </p>
              </div>

              <div className="hero-model">
                <span>
                  BEST PERFORMING MODEL
                </span>

                <strong>
                  {summary.best_mase_model}
                </strong>

                <small>
                  {summary.best_mase_category} ·
                  lowest MASE
                </small>
              </div>
            </section>

            {/* KPI GRID */}

            <section className="kpi-grid">
              <div className="kpi-card">
                <span>Total Categories</span>

                <strong>
                  {summary.total_categories}
                </strong>

                <small>
                  Tracked pharmaceutical categories
                </small>
              </div>

              <div className="kpi-card">
                <span>Forecast Demand</span>

                <strong>
                  {formatNumber(
                    summary.total_forecast_demand
                  )}
                </strong>

                <small>
                  Next{" "}
                  {summary.forecast_horizon_days} days
                </small>
              </div>

              <div className="kpi-card">
                <span>Recent Demand</span>

                <strong>
                  {formatNumber(
                    summary.total_recent_30d_demand
                  )}
                </strong>

                <small>
                  Previous 30 days
                </small>
              </div>

              <div className="kpi-card alert-kpi">
                <span>Categories Flagged</span>

                <strong>
                  {summary.flagged_categories}
                </strong>

                <small>
                  Require attention
                </small>
              </div>
            </section>

            {/* CHART + ALERTS */}

            <section className="content-grid">
              <div className="panel chart-panel">
                <div className="panel-header">
                  <div>
                    <span className="panel-label">
                      FORECAST
                    </span>

                    <h3>
                      Category demand outlook
                    </h3>
                  </div>

                  <select
                    value={selectedCategory}
                    onChange={(event) =>
                      setSelectedCategory(
                        event.target.value
                      )
                    }
                  >
                    {categories.map((category) => (
                      <option
                        key={category.category}
                        value={category.category}
                      >
                        {category.category}
                      </option>
                    ))}
                  </select>
                </div>

                {/* CHART META */}

                {selectedCategoryData && (
                  <div className="chart-meta">
                    <div>
                      <span>Model</span>
                      <strong>
                        {selectedCategoryData.model}
                      </strong>
                    </div>

                    <div>
                      <span>Recent 30D</span>

                      <strong>
                        {selectedCategoryData.recent_30d_mean.toFixed(
                          2
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Forecast 30D</span>

                      <strong>
                        {selectedCategoryData.forecast_30d_mean.toFixed(
                          2
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Change</span>

                      <strong
                        className={
                          selectedCategoryData.forecast_change_pct >=
                          0
                            ? "positive"
                            : "negative"
                        }
                      >
                        {formatPercent(
                          selectedCategoryData.forecast_change_pct
                        )}
                      </strong>
                    </div>
                  </div>
                )}

                {/* CHART */}

                <div className="chart">
                  <ResponsiveContainer
                    width="100%"
                    height={360}
                  >
                    <LineChart
                      data={forecast}
                      margin={{
                        top: 15,
                        right: 20,
                        left: 20,
                        bottom: 35,
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        vertical={false}
                        stroke={
                          chartColors.grid
                        }
                      />

                      <XAxis
                        dataKey="datum"
                        tick={{
                          fontSize: 11,
                          fill: chartColors.axis,
                        }}
                        tickFormatter={(value) =>
                          String(value).slice(5)
                        }
                        axisLine={{
                          stroke:
                            chartColors.grid,
                        }}
                        tickLine={false}
                        label={{
                          value: "Date",
                          position:
                            "insideBottom",
                          offset: -20,
                          fill:
                            chartColors.axis,
                          fontSize: 12,
                        }}
                      />

                      <YAxis
                        tick={{
                          fontSize: 11,
                          fill: chartColors.axis,
                        }}
                        axisLine={{
                          stroke:
                            chartColors.grid,
                        }}
                        tickLine={false}
                        label={{
                          value:
                            "Demand (units/day)",
                          angle: -90,
                          position:
                            "insideLeft",
                          offset: -5,
                          fill:
                            chartColors.axis,
                          fontSize: 12,
                        }}
                      />

                      <Tooltip
                        contentStyle={{
                          backgroundColor:
                            chartColors.tooltipBackground,
                          border: `1px solid ${chartColors.tooltipBorder}`,
                          borderRadius:
                            "8px",
                          color:
                            chartColors.tooltipText,
                        }}
                        labelStyle={{
                          color:
                            chartColors.tooltipText,
                        }}
                        formatter={(
                          value
                        ) => [
                          Number(value).toFixed(
                            2
                          ),
                          "Forecast demand",
                        ]}
                      />

                      <Line
                        type="monotone"
                        dataKey="forecast"
                        stroke={
                          chartColors.line
                        }
                        strokeWidth={3}
                        dot={false}
                        name="Forecast"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* ACTIVE ALERTS */}

              <div className="panel">
                <div className="panel-header">
                  <div>
                    <span className="panel-label">
                      ATTENTION
                    </span>

                    <h3>
                      Active alerts
                    </h3>
                  </div>

                  <button
                    className="text-button"
                    onClick={() =>
                      setPage("alerts")
                    }
                  >
                    View all
                    <ChevronRight size={15} />
                  </button>
                </div>

                <div className="alert-list">
                  {alerts.map((alert) => (
                    <div
                      className="alert-item"
                      key={alert.category}
                    >
                      <div className="alert-icon">
                        <AlertTriangle
                          size={17}
                        />
                      </div>

                      <div className="alert-content">
                        <div>
                          <strong>
                            {alert.category}
                          </strong>

                          <StatusBadge
                            status={
                              alert.status
                            }
                          />
                        </div>

                        <p>
                          {
                            alert.business_interpretation
                          }
                        </p>

                        <span>
                          Forecast change:{" "}
                          <b>
                            {formatPercent(
                              alert.forecast_change_pct
                            )}
                          </b>
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* CATEGORY PERFORMANCE */}

            <section className="panel">
              <div className="panel-header">
                <div>
                  <span className="panel-label">
                    CATEGORY ANALYSIS
                  </span>

                  <h3>
                    Forecast performance
                  </h3>
                </div>

                <button
                  className="text-button"
                  onClick={() =>
                    setPage("categories")
                  }
                >
                  Full analysis
                  <ChevronRight size={15} />
                </button>
              </div>

              <CategoryTable
                categories={categories}
              />
            </section>
          </>
        )}

        {/* =================================================
            FORECASTS PAGE
           ================================================= */}

        {page === "forecasts" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <span className="panel-label">
                  TIME SERIES
                </span>

                <h3>
                  30-day demand forecast
                </h3>
              </div>

              <select
                value={selectedCategory}
                onChange={(event) =>
                  setSelectedCategory(
                    event.target.value
                  )
                }
              >
                {categories.map((category) => (
                  <option
                    key={category.category}
                    value={category.category}
                  >
                    {category.category}
                  </option>
                ))}
              </select>
            </div>

            {/* FORECAST SUMMARY */}

            <div className="forecast-summary">
              {selectedCategoryData && (
                <>
                  <div>
                    <span>
                      Selected category
                    </span>

                    <strong>
                      {selectedCategory}
                    </strong>
                  </div>

                  <div>
                    <span>Model</span>

                    <strong>
                      {selectedCategoryData.model}
                    </strong>
                  </div>

                  <div>
                    <span>MASE</span>

                    <strong>
                      {selectedCategoryData.MASE.toFixed(
                        3
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>
                      Expected change
                    </span>

                    <strong
                      className={
                        selectedCategoryData.forecast_change_pct >=
                        0
                          ? "positive"
                          : "negative"
                      }
                    >
                      {formatPercent(
                        selectedCategoryData.forecast_change_pct
                      )}
                    </strong>
                  </div>
                </>
              )}
            </div>

            {/* LARGE FORECAST CHART */}

            <div className="large-chart">
              <ResponsiveContainer
                width="100%"
                height={500}
              >
                <LineChart
                  data={forecast}
                  margin={{
                    top: 20,
                    right: 25,
                    left: 25,
                    bottom: 45,
                  }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    stroke={
                      chartColors.grid
                    }
                  />

                  <XAxis
                    dataKey="datum"
                    tick={{
                      fontSize: 11,
                      fill: chartColors.axis,
                    }}
                    tickFormatter={(value) =>
                      String(value).slice(5)
                    }
                    axisLine={{
                      stroke:
                        chartColors.grid,
                    }}
                    tickLine={false}
                    label={{
                      value:
                        "Date",
                      position:
                        "insideBottom",
                      offset: -25,
                      fill:
                        chartColors.axis,
                      fontSize: 13,
                    }}
                  />

                  <YAxis
                    tick={{
                      fontSize: 11,
                      fill: chartColors.axis,
                    }}
                    axisLine={{
                      stroke:
                        chartColors.grid,
                    }}
                    tickLine={false}
                    label={{
                      value:
                        "Demand (units/day)",
                      angle: -90,
                      position:
                        "insideLeft",
                      offset: 0,
                      fill:
                        chartColors.axis,
                      fontSize: 13,
                    }}
                  />

                  <Tooltip
                    contentStyle={{
                      backgroundColor:
                        chartColors.tooltipBackground,
                      border: `1px solid ${chartColors.tooltipBorder}`,
                      borderRadius:
                        "8px",
                      color:
                        chartColors.tooltipText,
                    }}
                    labelStyle={{
                      color:
                        chartColors.tooltipText,
                    }}
                    formatter={(
                      value
                    ) => [
                      Number(value).toFixed(
                        2
                      ),
                      "Forecast demand",
                    ]}
                  />

                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke={
                      chartColors.line
                    }
                    strokeWidth={3}
                    dot={false}
                    name="Forecast demand"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        {/* =================================================
            CATEGORIES PAGE
           ================================================= */}

        {page === "categories" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <span className="panel-label">
                  PORTFOLIO
                </span>

                <h3>
                  Category-level forecasting
                  performance
                </h3>
              </div>
            </div>

            <CategoryTable
              categories={categories}
              large
            />
          </section>
        )}

        {/* =================================================
            MODELS PAGE
           ================================================= */}

        {page === "models" && (
          <>
            <section className="kpi-grid">
              <div className="kpi-card">
                <span>Categories</span>

                <strong>
                  {bestModels.length}
                </strong>

                <small>
                  Production forecast models
                </small>
              </div>

              <div className="kpi-card">
                <span>SARIMA</span>

                <strong>
                  {modelCounts.SARIMA ?? 0}
                </strong>

                <small>
                  Selected for production
                </small>
              </div>

              <div className="kpi-card">
                <span>LightGBM</span>

                <strong>
                  {modelCounts.LightGBM ?? 0}
                </strong>

                <small>
                  Selected for production
                </small>
              </div>

              <div className="kpi-card">
                <span>Best MASE</span>

                <strong>
                  {bestOverallModel
                    ? bestOverallModel.MASE.toFixed(
                        3
                      )
                    : "—"}
                </strong>

                <small>
                  {bestOverallModel
                    ? `${bestOverallModel.category} · ${bestOverallModel.model}`
                    : "No model data"}
                </small>
              </div>
            </section>

            <section className="panel page-panel">
              <div className="panel-header">
                <div>
                  <span className="panel-label">
                    MODEL BENCHMARK
                  </span>

                  <h3>
                    Forecasting model comparison
                  </h3>
                </div>
              </div>

              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Model</th>
                      <th>MAE</th>
                      <th>RMSE</th>
                      <th>sMAPE</th>
                      <th>WAPE</th>
                      <th>MASE</th>
                    </tr>
                  </thead>

                  <tbody>
                    {models.map(
                      (model, index) => (
                        <tr
                          key={`${model.category}-${model.model}-${index}`}
                        >
                          <td>
                            <strong>
                              {model.category}
                            </strong>
                          </td>

                          <td>
                            {model.model}
                          </td>

                          <td>
                            {model.MAE.toFixed(
                              2
                            )}
                          </td>

                          <td>
                            {model.RMSE.toFixed(
                              2
                            )}
                          </td>

                          <td>
                            {model.sMAPE.toFixed(
                              1
                            )}
                            %
                          </td>

                          <td>
                            {model.WAPE.toFixed(
                              1
                            )}
                            %
                          </td>

                          <td>
                            <strong>
                              {model.MASE.toFixed(
                                3
                              )}
                            </strong>
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {/* =================================================
            ALERTS PAGE
           ================================================= */}

        {page === "alerts" && (
          <section className="panel page-panel">
            <div className="panel-header">
              <div>
                <span className="panel-label">
                  REQUIRES ATTENTION
                </span>

                <h3>
                  Forecast alerts
                </h3>
              </div>

              <span className="alert-count">
                {alerts.length} active
              </span>
            </div>

            <div className="alert-grid">
              {alerts.map((alert) => (
                <article
                  className="large-alert"
                  key={alert.category}
                >
                  <div className="large-alert-header">
                    <div className="alert-icon">
                      <AlertTriangle
                        size={20}
                      />
                    </div>

                    <div>
                      <h3>
                        {alert.category}
                      </h3>

                      <StatusBadge
                        status={
                          alert.status
                        }
                      />
                    </div>
                  </div>

                  <p>
                    {
                      alert.business_interpretation
                    }
                  </p>

                  <div className="alert-metrics">
                    <div>
                      <span>
                        Recent 30D
                      </span>

                      <strong>
                        {alert.recent_30d_mean.toFixed(
                          2
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>
                        Forecast 30D
                      </span>

                      <strong>
                        {alert.forecast_30d_mean.toFixed(
                          2
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Change</span>

                      <strong
                        className={
                          alert.forecast_change_pct >=
                          0
                            ? "positive"
                            : "negative"
                        }
                      >
                        {formatPercent(
                          alert.forecast_change_pct
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>MASE</span>

                      <strong>
                        {alert.MASE.toFixed(
                          3
                        )}
                      </strong>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

/*
 * =========================================================
 * CATEGORY TABLE
 * =========================================================
 */

function CategoryTable({
  categories,
  large = false,
}: {
  categories: Category[];
  large?: boolean;
}) {
  return (
    <div
      className={`table-wrapper ${
        large ? "large-table" : ""
      }`}
    >
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Recent 30D</th>
            <th>Forecast 30D</th>
            <th>Change</th>
            <th>MASE</th>
            <th>Status</th>
          </tr>
        </thead>

        <tbody>
          {categories.map((category) => (
            <tr key={category.category}>
              <td>
                <strong>
                  {category.category}
                </strong>
              </td>

              <td>
                {category.model}
              </td>

              <td>
                {category.recent_30d_mean.toFixed(
                  2
                )}
              </td>

              <td>
                {category.forecast_30d_mean.toFixed(
                  2
                )}
              </td>

              <td>
                <span
                  className={
                    category.forecast_change_pct >=
                    0
                      ? "positive"
                      : "negative"
                  }
                >
                  {category.forecast_change_pct >=
                  0 ? (
                    <TrendingUp
                      size={14}
                    />
                  ) : (
                    <TrendingDown
                      size={14}
                    />
                  )}

                  {formatPercent(
                    category.forecast_change_pct
                  )}
                </span>
              </td>

              <td>
                <strong>
                  {category.MASE.toFixed(
                    3
                  )}
                </strong>
              </td>

              <td>
                <StatusBadge
                  status={
                    category.status
                  }
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;