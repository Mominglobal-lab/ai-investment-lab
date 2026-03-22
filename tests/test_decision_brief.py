from __future__ import annotations

import json
from pathlib import Path

from reports.decision_brief import generate_decision_brief


def _sample_simulation_result() -> dict:
    return {
        "metadata": {
            "run_timestamp": "2026-03-05T15:00:00+00:00",
            "period_start": "2021-03-05",
            "period_end": "2026-03-05",
            "rebalance_rule": "monthly",
            "benchmark_ticker": "SPY",
            "simulation_mode": "monte_carlo",
        },
        "portfolio": {
            "holdings": [
                {"ticker": "AAPL", "weight": 0.50},
                {"ticker": "MSFT", "weight": 0.30},
                {"ticker": "NVDA", "weight": 0.20},
            ],
            "dropped_tickers": [],
            "warnings": [],
            "weighting_mode": "Market cap weight",
        },
        "summary": {
            "CAGR": 0.12,
            "volatility": 0.24,
            "Sharpe_ratio": 0.50,
            "max_drawdown": -0.28,
            "worst_day": -0.05,
            "worst_month": -0.12,
            "VaR_95": -0.025,
            "CVaR_95": -0.040,
            "correlation_with_benchmark": 0.90,
            "beta_relative_to_benchmark": 1.30,
        },
        "scenario_results": {
            "ending_value_percentiles": {"p05": 8500.0, "p25": 9800.0, "p50": 11200.0, "p75": 12800.0, "p95": 15600.0},
            "max_drawdown_percentiles": {"p05": -0.45, "p25": -0.32, "p50": -0.24, "p75": -0.18, "p95": -0.10},
            "probability_of_loss": 0.41,
        },
    }


def test_generate_decision_brief_creates_json_and_html(tmp_path):
    sim = _sample_simulation_result()
    out = generate_decision_brief(simulation_result=sim, output_dir=str(tmp_path), format="html")

    json_path = Path(out["json_path"])
    html_path = Path(out["html_path"])
    assert json_path.exists()
    assert html_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "decision_insights" in data
    assert isinstance(data["decision_insights"], list)
    assert len(data["decision_insights"]) > 0


def test_decision_insights_are_deterministic_for_same_input(tmp_path):
    sim = _sample_simulation_result()
    out1 = generate_decision_brief(simulation_result=sim, output_dir=str(tmp_path), format="html")
    out2 = generate_decision_brief(simulation_result=sim, output_dir=str(tmp_path), format="html")

    d1 = json.loads(Path(out1["json_path"]).read_text(encoding="utf-8"))
    d2 = json.loads(Path(out2["json_path"]).read_text(encoding="utf-8"))
    assert d1["decision_insights"] == d2["decision_insights"]


def test_generate_decision_brief_escapes_html_content(tmp_path):
    sim = _sample_simulation_result()
    sim["portfolio"]["holdings"][0]["ticker"] = '<script>alert("x")</script>'
    sim["metadata"]["benchmark_ticker"] = '<b>QQQ</b>'
    title = '<img src=x onerror=alert("x")>'

    out = generate_decision_brief(simulation_result=sim, output_dir=str(tmp_path), format="html", title=title)
    html_text = Path(out["html_path"]).read_text(encoding="utf-8")

    assert '<script>alert("x")</script>' not in html_text
    assert '<b>QQQ</b>' not in html_text
    assert '<img src=x onerror=alert("x")>' not in html_text
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html_text
    assert "&lt;b&gt;QQQ&lt;/b&gt;" in html_text
    assert "&lt;img src=x onerror=alert(&quot;x&quot;)&gt;" in html_text

