from __future__ import annotations

import pandas as pd

from data_pipeline.run_pipeline import run_prices_cache_pipeline


class DummyPriceRefreshResult:
    def __init__(self, requested_count: int):
        self.data = pd.DataFrame(columns=["Ticker", "Date", "AdjClose", "Close", "Volume"])
        self.requested_count = requested_count
        self.success_count = 0
        self.failure_count = requested_count
        self.errors_sample = []


def test_prices_pipeline_always_includes_core_benchmarks(tmp_path, monkeypatch):
    import data_pipeline.run_pipeline as rp

    seen_tickers: list[str] = []

    def fake_refresh_prices_yfinance(tickers, lookback_years=5):
        seen_tickers.extend(tickers)
        return DummyPriceRefreshResult(requested_count=len(tickers))

    monkeypatch.setattr(rp, "refresh_prices_yfinance", fake_refresh_prices_yfinance)

    cache_path = tmp_path / "prices.parquet"
    report_path = tmp_path / "prices_health.json"
    result = run_prices_cache_pipeline(
        prices_cache_path=str(cache_path),
        health_report_path=str(report_path),
        fundamentals_cache_paths=[str(tmp_path / "missing_fundamentals.parquet")],
        benchmark_ticker="QQQ",
        max_age_days=0.0,
    )

    assert result.requested_count >= 4
    assert {"SPY", "QQQ", "IWM", "DIA"}.issubset(set(seen_tickers))

