from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_builtin(value.item())
        except Exception:
            pass
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(_to_builtin(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def _derive_insights(simulation_result: dict[str, Any]) -> list[str]:
    insights: list[str] = []
    portfolio = simulation_result.get("portfolio", {}) or {}
    summary = simulation_result.get("summary", {}) or {}
    scenario = simulation_result.get("scenario_results")
    holdings = portfolio.get("holdings", []) or []

    if holdings:
        max_w = max(float(h.get("weight", 0.0)) for h in holdings)
        if max_w > 0.35:
            insights.append("Concentration risk is elevated: at least one holding exceeds 35% weight.")

    beta = float(summary.get("beta_relative_to_benchmark", float("nan")))
    if beta == beta and beta > 1.2:
        insights.append("High beta warning: portfolio beta is above 1.2 and may amplify benchmark moves.")

    mdd = float(summary.get("max_drawdown", float("nan")))
    if mdd == mdd:
        if mdd <= -0.35:
            insights.append("Drawdown severity: severe (max drawdown worse than -35%).")
        elif mdd <= -0.20:
            insights.append("Drawdown severity: moderate (max drawdown between -20% and -35%).")
        else:
            insights.append("Drawdown severity: contained (max drawdown better than -20%).")

    corr = float(summary.get("correlation_with_benchmark", float("nan")))
    if corr == corr and corr > 0.85:
        insights.append("Benchmark dependence is high: correlation with benchmark exceeds 0.85.")

    if scenario:
        p_loss = float(scenario.get("probability_of_loss", float("nan")))
        if p_loss == p_loss and p_loss > 0.30:
            insights.append("Downside probability alert: Monte Carlo probability of loss exceeds 30%.")

    if not insights:
        insights.append("Risk profile appears balanced under the current assumptions and historical sample.")
    return insights


def _artifact_rel_paths(run_id: str) -> dict[str, str]:
    return {
        "json": f"{run_id}/decision_brief.json",
        "html": f"{run_id}/decision_brief.html",
    }


def _html_table_from_pairs(pairs: list[tuple[str, Any]]) -> str:
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in pairs)
    return f"<table>{rows}</table>"


def _html_holdings(holdings: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{h.get('ticker')}</td>"
        f"<td>{float(h.get('weight', 0.0)):.4f}</td>"
        "</tr>"
        for h in holdings
    )
    return (
        "<table>"
        "<thead><tr><th>Ticker</th><th>Weight</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def _html_metrics(summary: dict[str, Any]) -> str:
    preferred = [
        "CAGR",
        "volatility",
        "Sharpe_ratio",
        "max_drawdown",
        "VaR_95",
        "CVaR_95",
        "worst_day",
        "worst_month",
        "correlation_with_benchmark",
        "beta_relative_to_benchmark",
    ]
    pairs = []
    for k in preferred:
        if k in summary:
            v = summary.get(k)
            if isinstance(v, (int, float)):
                pairs.append((k, f"{float(v):.6f}"))
            else:
                pairs.append((k, v))
    return _html_table_from_pairs(pairs)


def _html_scenario(scenario: dict[str, Any] | None) -> str:
    if not scenario:
        return "<p>Monte Carlo scenario not requested.</p>"
    ev = scenario.get("ending_value_percentiles", {}) or {}
    dd = scenario.get("max_drawdown_percentiles", {}) or {}
    p_loss = scenario.get("probability_of_loss")
    rows = [
        ("Ending Value p05", ev.get("p05")),
        ("Ending Value p50", ev.get("p50")),
        ("Ending Value p95", ev.get("p95")),
        ("Max Drawdown p05", dd.get("p05")),
        ("Max Drawdown p50", dd.get("p50")),
        ("Max Drawdown p95", dd.get("p95")),
        ("Probability of Loss", p_loss),
    ]
    return _html_table_from_pairs([(k, f"{float(v):.6f}" if isinstance(v, (int, float)) else v) for k, v in rows])


def generate_decision_brief(
    simulation_result,
    output_dir="data/run_artifacts",
    format="html",
    title="Portfolio Decision Brief",
):
    if format not in {"html"}:
        raise ValueError("Only html export is supported")

    sim = _to_builtin(simulation_result or {})
    metadata = dict(sim.get("metadata", {}) or {})
    portfolio = dict(sim.get("portfolio", {}) or {})
    summary = dict(sim.get("summary", {}) or {})
    scenario = sim.get("scenario_results")

    run_seed = {
        "metadata": metadata,
        "portfolio": portfolio,
        "summary": summary,
        "scenario": scenario,
        "title": title,
    }
    run_hash = _hash_payload(run_seed)
    run_ts = metadata.get("run_timestamp") or _now_iso()
    run_dt = datetime.fromisoformat(str(run_ts).replace("Z", "+00:00"))
    run_id = f"{run_dt.strftime('%Y%m%dT%H%M%SZ')}_{run_hash}"

    export_ts = _now_iso()
    out_root = Path(output_dir)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    insights = _derive_insights(sim)
    files = _artifact_rel_paths(run_id)
    brief = {
        "metadata": {
            **metadata,
            "run_id": run_id,
            "export_timestamp": export_ts,
            "title": title,
        },
        "portfolio": portfolio,
        "summary": summary,
        "scenario": scenario,
        "decision_insights": insights,
        "files": files,
    }
    brief = _to_builtin(brief)

    json_path = run_dir / "decision_brief.json"
    html_path = run_dir / "decision_brief.html"
    json_path.write_text(json.dumps(brief, indent=2, sort_keys=False), encoding="utf-8")

    used_artifacts = [
        "data/prices_cache.parquet",
        "data/prices_health_report.json",
    ]
    if portfolio.get("weighting_mode") == "Market cap weight":
        used_artifacts.append("data/fundamentals_cache.parquet")

    metadata_pairs = [
        ("Run ID", brief["metadata"].get("run_id")),
        ("Export Timestamp", brief["metadata"].get("export_timestamp")),
        ("Simulation Mode", brief["metadata"].get("simulation_mode")),
        ("Period Start", brief["metadata"].get("period_start")),
        ("Period End", brief["metadata"].get("period_end")),
        ("Rebalance Rule", brief["metadata"].get("rebalance_rule")),
        ("Benchmark", brief["metadata"].get("benchmark_ticker")),
    ]
    risk_pairs = [
        ("VaR 95%", summary.get("VaR_95")),
        ("CVaR 95%", summary.get("CVaR_95")),
        ("Max Drawdown", summary.get("max_drawdown")),
        ("Worst Day", summary.get("worst_day")),
        ("Worst Month", summary.get("worst_month")),
    ]
    risk_pairs_fmt = [(k, f"{float(v):.6f}" if isinstance(v, (int, float)) else v) for k, v in risk_pairs]
    insights_html = "".join(f"<li>{x}</li>" for x in insights)
    artifacts_html = "".join(f"<li>{a}</li>" for a in used_artifacts)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 28px; color: #0c1423; background: #f8fbff; }}
h1, h2 {{ margin-bottom: 8px; }}
section {{ background: #ffffff; border: 1px solid #d5e2f3; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #d8e2f0; padding: 7px 8px; text-align: left; }}
th {{ background: #eff5fd; font-weight: 700; }}
ul {{ margin: 8px 0 0 18px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<section><h2>Run Metadata</h2>{_html_table_from_pairs(metadata_pairs)}</section>
<section><h2>Holdings</h2>{_html_holdings(portfolio.get("holdings", []) or [])}</section>
<section><h2>Key Metrics</h2>{_html_metrics(summary)}</section>
<section><h2>Risk</h2>{_html_table_from_pairs(risk_pairs_fmt)}</section>
<section><h2>Monte Carlo</h2>{_html_scenario(scenario)}</section>
<section><h2>Decision Insights</h2><ul>{insights_html}</ul></section>
<section><h2>Reproducibility</h2><ul>{artifacts_html}</ul></section>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }

