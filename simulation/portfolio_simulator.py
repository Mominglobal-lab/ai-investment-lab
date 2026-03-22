from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


PRICE_CACHE_PATH = "data/prices_cache.parquet"
PRICE_SCHEMA_COLUMNS = ["Ticker", "Date", "AdjClose", "Close", "Volume"]
TRADING_DAYS_PER_YEAR = 252.0
INITIAL_CAPITAL = 10_000.0


@dataclass(frozen=True)
class NormalizedHoldings:
    weights: dict[str, float]
    dropped_tickers: list[str]
    warnings: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _iso_dates(index: pd.Index) -> list[str]:
    dt_idx = pd.to_datetime(index, errors="coerce")
    return [d.strftime("%Y-%m-%d") for d in dt_idx if pd.notna(d)]


def _load_prices_cache(path: str = PRICE_CACHE_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in PRICE_SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"prices cache schema mismatch; missing columns: {', '.join(missing)}")
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["AdjClose"] = pd.to_numeric(out["AdjClose"], errors="coerce")
    out["Close"] = pd.to_numeric(out.get("Close"), errors="coerce")
    out["Volume"] = pd.to_numeric(out.get("Volume"), errors="coerce")
    out = out.dropna(subset=["Ticker", "Date", "AdjClose"])
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _normalize_input_holdings(holdings: Iterable[tuple[str, float]]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for ticker, weight in holdings:
        t = str(ticker or "").strip().upper()
        if not t:
            continue
        w = _as_float(weight, default=0.0)
        if w < 0:
            raise ValueError("Negative weights are not allowed")
        clean[t] = clean.get(t, 0.0) + w
    if not clean:
        raise ValueError("No valid holdings provided")
    total = float(sum(clean.values()))
    if total <= 0:
        raise ValueError("Holdings must have positive total weight")
    return {k: float(v / total) for k, v in clean.items()}


def _normalize_available_holdings(
    requested_weights: dict[str, float],
    available_tickers: set[str],
    strict: bool,
) -> NormalizedHoldings:
    dropped = [t for t in requested_weights if t not in available_tickers]
    if strict and dropped:
        raise ValueError(f"Missing ticker data in price cache: {', '.join(dropped)}")
    kept = {t: w for t, w in requested_weights.items() if t in available_tickers}
    if not kept:
        raise ValueError("None of the requested holdings have available price data")
    total = float(sum(kept.values()))
    renorm = {t: float(w / total) for t, w in kept.items()}
    warnings: list[str] = []
    if dropped:
        warnings.append(f"Dropped tickers with missing data: {', '.join(dropped)}")
    if abs(total - 1.0) > 1e-12:
        warnings.append("Weights were renormalized after dropping unavailable tickers")
    return NormalizedHoldings(weights=renorm, dropped_tickers=dropped, warnings=warnings)


def _portfolio_returns_with_rebalance(
    asset_returns: pd.DataFrame, weights: dict[str, float], rebalance_rule: str, initial_capital: float
) -> tuple[pd.Series, pd.Series]:
    target = pd.Series(weights, dtype=float).reindex(asset_returns.columns).fillna(0.0)
    target = target / target.sum()

    values = target * initial_capital
    previous_total = float(values.sum())
    portfolio_returns: list[float] = []
    portfolio_values: list[float] = []
    last_rebalanced_month: str | None = None

    for dt, row in asset_returns.iterrows():
        month_key = pd.Timestamp(dt).strftime("%Y-%m")
        if rebalance_rule == "monthly" and month_key != last_rebalanced_month:
            values = target * previous_total
            last_rebalanced_month = month_key
        elif rebalance_rule == "none" and last_rebalanced_month is None:
            last_rebalanced_month = month_key

        values = values * (1.0 + row.fillna(0.0))
        current_total = float(values.sum())
        daily_ret = (current_total / previous_total) - 1.0 if previous_total != 0 else 0.0
        portfolio_returns.append(float(daily_ret))
        portfolio_values.append(float(current_total))
        previous_total = current_total

    ret = pd.Series(portfolio_returns, index=asset_returns.index, name="portfolio_return")
    val = pd.Series(portfolio_values, index=asset_returns.index, name="portfolio_value")
    return ret, val


def _compute_drawdown(value_series: pd.Series) -> pd.Series:
    running_max = value_series.cummax()
    return (value_series / running_max) - 1.0


def _compute_summary_metrics(
    portfolio_returns: pd.Series,
    portfolio_values: pd.Series,
    drawdown: pd.Series,
    benchmark_returns: pd.Series | None,
    risk_free_rate: float,
    initial_capital: float,
) -> dict[str, float]:
    if portfolio_returns.empty or portfolio_values.empty:
        raise ValueError("Insufficient return history to compute metrics")

    years = max(len(portfolio_returns) / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
    end_value = float(portfolio_values.iloc[-1])
    cagr = (end_value / initial_capital) ** (1.0 / years) - 1.0
    vol = float(portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float((cagr - risk_free_rate) / vol) if vol > 0 else float("nan")
    max_drawdown = float(drawdown.min())
    worst_day = float(portfolio_returns.min())

    monthly = (1.0 + portfolio_returns).groupby(portfolio_returns.index.to_period("M")).prod() - 1.0
    worst_month = float(monthly.min()) if not monthly.empty else float("nan")

    var95 = float(np.quantile(portfolio_returns.values, 0.05))
    cvar_tail = portfolio_returns[portfolio_returns <= var95]
    cvar95 = float(cvar_tail.mean()) if not cvar_tail.empty else float(var95)

    corr = float("nan")
    beta = float("nan")
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([portfolio_returns.rename("p"), benchmark_returns.rename("b")], axis=1).dropna()
        if len(aligned) >= 2:
            p_std = float(aligned["p"].std(ddof=1))
            b_std = float(aligned["b"].std(ddof=1))
            if p_std > 0 and b_std > 0:
                corr = float(aligned["p"].corr(aligned["b"]))
            bvar = float(aligned["b"].var(ddof=1))
            if bvar > 0:
                beta = float(aligned["p"].cov(aligned["b"]) / bvar)

    return {
        "CAGR": float(cagr),
        "volatility": float(vol),
        "Sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "worst_day": float(worst_day),
        "worst_month": float(worst_month),
        "VaR_95": float(var95),
        "CVaR_95": float(cvar95),
        "correlation_with_benchmark": float(corr),
        "beta_relative_to_benchmark": float(beta),
    }


def _monte_carlo_scenarios(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    monte_carlo_paths: int,
    horizon_days: int,
    initial_capital: float,
) -> dict[str, float | dict[str, float]]:
    if asset_returns.empty:
        raise ValueError("Insufficient data for Monte Carlo simulation")

    w = pd.Series(weights, dtype=float).reindex(asset_returns.columns).fillna(0.0).values
    mu = asset_returns.mean().values
    cov = asset_returns.cov().values
    rng = np.random.default_rng(42)

    ending_values: list[float] = []
    max_drawdowns: list[float] = []
    for _ in range(int(monte_carlo_paths)):
        draws = rng.multivariate_normal(mean=mu, cov=cov, size=int(horizon_days))
        p_ret = draws @ w
        path_values = initial_capital * np.cumprod(1.0 + p_ret)
        peak = np.maximum.accumulate(path_values)
        dd = (path_values / peak) - 1.0
        ending_values.append(float(path_values[-1]))
        max_drawdowns.append(float(np.min(dd)))

    end_arr = np.array(ending_values, dtype=float)
    dd_arr = np.array(max_drawdowns, dtype=float)
    probs = float(np.mean(end_arr < initial_capital))

    def pct_map(arr: np.ndarray) -> dict[str, float]:
        return {
            "p05": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "ending_value_percentiles": pct_map(end_arr),
        "max_drawdown_percentiles": pct_map(dd_arr),
        "probability_of_loss": probs,
    }


def _decision_insights(
    summary: dict[str, float],
    weights: dict[str, float],
    scenario: dict | None,
    ending_portfolio_value: float,
    ending_benchmark_value: float | None,
    initial_capital: float,
) -> list[str]:
    insights: list[str] = []
    portfolio_gain_pct = ((ending_portfolio_value / initial_capital) - 1.0) * 100.0
    if ending_benchmark_value is not None:
        benchmark_gain_pct = ((ending_benchmark_value / initial_capital) - 1.0) * 100.0
        rel_gap = ending_portfolio_value - ending_benchmark_value
        direction = "outperformed" if rel_gap >= 0 else "underperformed"
        insights.append(
            f"Portfolio grew from ${initial_capital:,.0f} to ${ending_portfolio_value:,.0f} ({portfolio_gain_pct:.1f}%), "
            f"vs benchmark ${ending_benchmark_value:,.0f} ({benchmark_gain_pct:.1f}%)."
        )
        insights.append(f"Portfolio {direction} by ${abs(rel_gap):,.0f}.")
    else:
        insights.append(
            f"Portfolio grew from ${initial_capital:,.0f} to ${ending_portfolio_value:,.0f} ({portfolio_gain_pct:.1f}%)."
        )

    hhi = float(sum(w * w for w in weights.values()))
    if hhi >= 0.35:
        insights.append("Portfolio concentration risk is elevated (weight concentration is high).")
    elif hhi >= 0.25:
        insights.append("Portfolio concentration is moderate; diversification could reduce single-name risk.")
    else:
        insights.append("Portfolio concentration appears diversified based on current weights.")

    vol = summary.get("volatility", float("nan"))
    if np.isfinite(vol) and vol > 0.30:
        insights.append("Annualized volatility is high and may exceed conservative risk tolerance.")
    elif np.isfinite(vol) and vol > 0.20:
        insights.append("Annualized volatility is moderate to high.")

    cvar = summary.get("CVaR_95", float("nan"))
    if np.isfinite(cvar) and cvar < -0.03:
        insights.append("Downside tail risk is meaningful (CVaR 95% below -3% daily).")

    if scenario is not None:
        p_loss = _as_float(scenario.get("probability_of_loss"), default=float("nan"))
        if np.isfinite(p_loss) and p_loss > 0.4:
            insights.append("Monte Carlo scenario indicates elevated probability of capital loss over the chosen horizon.")

    return insights


def simulate_portfolio(
    holdings,
    lookback_years=5,
    rebalance_rule="none",
    benchmark="SPY",
    mode="historical",
    monte_carlo_paths=1000,
    horizon_days=252,
    risk_free_rate=0,
    strict=False,
    initial_capital=10_000,
):
    if rebalance_rule not in {"none", "monthly"}:
        raise ValueError("rebalance_rule must be one of: 'none', 'monthly'")
    if mode not in {"historical", "monte_carlo"}:
        raise ValueError("mode must be one of: 'historical', 'monte_carlo'")
    initial_capital = float(initial_capital)
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be a positive number")

    requested_weights = _normalize_input_holdings(holdings)
    px = _load_prices_cache(PRICE_CACHE_PATH)
    max_dt = px["Date"].max()
    min_dt = max_dt - pd.DateOffset(years=int(lookback_years))
    px = px[(px["Date"] >= min_dt) & (px["Date"] <= max_dt)].copy()
    if px.empty:
        raise ValueError("No price data available in the requested lookback window")

    pivot = px.pivot_table(index="Date", columns="Ticker", values="AdjClose", aggfunc="last").sort_index()
    available = set([str(c).upper() for c in pivot.columns])
    normalized = _normalize_available_holdings(requested_weights, available, strict=bool(strict))
    weights = normalized.weights

    holdings_prices = pivot[list(weights.keys())].dropna(how="any")
    if holdings_prices.shape[0] < 2:
        raise ValueError("Insufficient aligned price history after ticker/date alignment")

    asset_returns = holdings_prices.pct_change().dropna(how="any")
    if asset_returns.empty:
        raise ValueError("Unable to compute portfolio returns from available price history")

    portfolio_returns, portfolio_values = _portfolio_returns_with_rebalance(
        asset_returns=asset_returns,
        weights=weights,
        rebalance_rule=rebalance_rule,
        initial_capital=initial_capital,
    )
    drawdown = _compute_drawdown(portfolio_values)

    benchmark_symbol = str(benchmark or "").strip().upper()
    benchmark_value: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    simulation_start = holdings_prices.index.min()
    simulation_end = holdings_prices.index.max()
    if benchmark_symbol and benchmark_symbol in pivot.columns:
        b_prices = pivot[benchmark_symbol].dropna()
        b_window = b_prices[(b_prices.index >= simulation_start) & (b_prices.index <= simulation_end)]
        b_window = b_window.reindex(holdings_prices.index).ffill().dropna()
        if len(b_window) >= 2:
            benchmark_returns = b_window.pct_change().dropna()
            benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).dropna()
            benchmark_value = initial_capital * (b_window / float(b_window.iloc[0]))
            benchmark_value = benchmark_value.reindex(portfolio_returns.index)

    summary = _compute_summary_metrics(
        portfolio_returns=portfolio_returns,
        portfolio_values=portfolio_values,
        drawdown=drawdown,
        benchmark_returns=benchmark_returns,
        risk_free_rate=float(risk_free_rate),
        initial_capital=initial_capital,
    )

    scenario = None
    if mode == "monte_carlo":
        scenario = _monte_carlo_scenarios(
            asset_returns=asset_returns,
            weights=weights,
            monte_carlo_paths=int(monte_carlo_paths),
            horizon_days=int(horizon_days),
            initial_capital=initial_capital,
        )

    end_portfolio_value = float(portfolio_values.iloc[-1])
    end_benchmark_value = float(benchmark_value.iloc[-1]) if benchmark_value is not None and not benchmark_value.empty else None

    result = {
        "metadata": {
            "run_timestamp": _utc_now_iso(),
            "period_start": simulation_start.strftime("%Y-%m-%d"),
            "period_end": simulation_end.strftime("%Y-%m-%d"),
            "rebalance_rule": rebalance_rule,
            "benchmark_ticker": benchmark_symbol,
            "simulation_mode": mode,
            "initial_capital": float(initial_capital),
        },
        "portfolio": {
            "holdings": [{"ticker": t, "weight": float(w)} for t, w in weights.items()],
            "dropped_tickers": list(normalized.dropped_tickers),
            "warnings": list(normalized.warnings),
        },
        "timeseries": {
            "dates": _iso_dates(portfolio_returns.index),
            "portfolio_value": [float(x) for x in portfolio_values.tolist()],
            "portfolio_returns": [float(x) for x in portfolio_returns.tolist()],
            "drawdown": [float(x) for x in drawdown.tolist()],
            "benchmark_value": [float(x) for x in benchmark_value.tolist()] if benchmark_value is not None else None,
        },
        "summary": {k: float(v) for k, v in summary.items()},
        "scenario_results": scenario,
        "decision_insights": _decision_insights(
            summary=summary,
            weights=weights,
            scenario=scenario,
            ending_portfolio_value=end_portfolio_value,
            ending_benchmark_value=end_benchmark_value,
            initial_capital=initial_capital,
        ),
    }
    return result
