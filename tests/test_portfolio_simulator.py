from __future__ import annotations

import pandas as pd
import pytest

from simulation.portfolio_simulator import simulate_portfolio


def _make_prices(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["Ticker", "Date", "AdjClose"])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = out["AdjClose"]
    out["Volume"] = 1_000_000
    return out[["Ticker", "Date", "AdjClose", "Close", "Volume"]]


def test_weight_normalization(monkeypatch):
    prices = _make_prices(
        [
            ("AAA", "2025-01-01", 100),
            ("AAA", "2025-01-02", 101),
            ("BBB", "2025-01-01", 100),
            ("BBB", "2025-01-02", 99),
            ("SPY", "2025-01-01", 100),
            ("SPY", "2025-01-02", 100),
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: prices.copy())

    res = simulate_portfolio([("AAA", 0.2), ("BBB", 0.2)], lookback_years=5, strict=True)
    weights = {h["ticker"]: h["weight"] for h in res["portfolio"]["holdings"]}
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
    assert pytest.approx(weights["AAA"], rel=1e-9) == 0.5
    assert pytest.approx(weights["BBB"], rel=1e-9) == 0.5


def test_drawdown_calculation(monkeypatch):
    prices = _make_prices(
        [
            ("AAA", "2025-01-01", 100),
            ("AAA", "2025-01-02", 120),
            ("AAA", "2025-01-03", 60),
            ("AAA", "2025-01-06", 60),
            ("SPY", "2025-01-01", 100),
            ("SPY", "2025-01-02", 100),
            ("SPY", "2025-01-03", 100),
            ("SPY", "2025-01-06", 100),
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: prices.copy())

    res = simulate_portfolio([("AAA", 1.0)], lookback_years=5, strict=True)
    max_dd = res["summary"]["max_drawdown"]
    assert pytest.approx(max_dd, rel=1e-9) == -0.5


def test_portfolio_return_computation(monkeypatch):
    prices = _make_prices(
        [
            ("AAA", "2025-01-01", 100),
            ("AAA", "2025-01-02", 110),
            ("BBB", "2025-01-01", 100),
            ("BBB", "2025-01-02", 100),
            ("SPY", "2025-01-01", 100),
            ("SPY", "2025-01-02", 100),
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: prices.copy())

    res = simulate_portfolio([("AAA", 0.5), ("BBB", 0.5)], lookback_years=5, strict=True)
    first_ret = res["timeseries"]["portfolio_returns"][0]
    assert pytest.approx(first_ret, rel=1e-9) == 0.05


def test_missing_ticker_handling(monkeypatch):
    prices = _make_prices(
        [
            ("AAA", "2025-01-01", 100),
            ("AAA", "2025-01-02", 101),
            ("SPY", "2025-01-01", 100),
            ("SPY", "2025-01-02", 100),
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: prices.copy())

    loose = simulate_portfolio([("AAA", 0.6), ("MISSING", 0.4)], strict=False)
    assert loose["portfolio"]["dropped_tickers"] == ["MISSING"]
    w = {h["ticker"]: h["weight"] for h in loose["portfolio"]["holdings"]}
    assert pytest.approx(w["AAA"], rel=1e-9) == 1.0

    with pytest.raises(ValueError):
        simulate_portfolio([("AAA", 0.6), ("MISSING", 0.4)], strict=True)


def test_monthly_rebalance_logic(monkeypatch):
    prices = _make_prices(
        [
            ("AAA", "2025-01-30", 100),
            ("AAA", "2025-01-31", 200),
            ("AAA", "2025-02-03", 200),
            ("AAA", "2025-02-04", 100),
            ("BBB", "2025-01-30", 100),
            ("BBB", "2025-01-31", 100),
            ("BBB", "2025-02-03", 100),
            ("BBB", "2025-02-04", 100),
            ("SPY", "2025-01-30", 100),
            ("SPY", "2025-01-31", 100),
            ("SPY", "2025-02-03", 100),
            ("SPY", "2025-02-04", 100),
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: prices.copy())

    none = simulate_portfolio([("AAA", 0.5), ("BBB", 0.5)], rebalance_rule="none", strict=True)
    monthly = simulate_portfolio([("AAA", 0.5), ("BBB", 0.5)], rebalance_rule="monthly", strict=True)
    end_none = none["timeseries"]["portfolio_value"][-1]
    end_monthly = monthly["timeseries"]["portfolio_value"][-1]

    assert pytest.approx(end_none, rel=1e-9) == 10000.0
    assert pytest.approx(end_monthly, rel=1e-9) == 11250.0

