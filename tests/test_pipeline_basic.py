from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.run_pipeline import run_stock_fundamentals_pipeline


class DummyRefreshResult:
    def __init__(self, df: pd.DataFrame, requested: int, success: int, failed: int, rate_limited: bool, errors: list[str]):
        self.data = df
        self.requested_count = requested
        self.success_count = success
        self.failure_count = failed
        self.rate_limited = rate_limited
        self.errors_sample = errors


def test_pipeline_writes_cache_and_report(tmp_path, monkeypatch):
    cache_path = tmp_path / "fundamentals.parquet"
    report_path = tmp_path / "health.json"

    # Import inside test so monkeypatch can target the module used by run_pipeline.
    import data_pipeline.run_pipeline as rp

    df = pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "Company": "Apple Inc.",
                "Sector": "Technology",
                "Close": 100.0,
                "MarketCap": 1_000_000_000_000.0,
                "Revenue_Growth_YoY_Pct": 10.0,
                "Earnings_Growth_Pct": 12.0,
                "EBITDA_Margin": 0.30,
                "ROE": 0.25,
                "PE_Ratio": 25.0,
                "PEG_Ratio": 2.0,
                "Rule_of_40": 40.0,
            }
        ]
    )

    # Fill any schema columns the local project expects without hardcoding them here.
    for col in rp.SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[rp.SCHEMA_COLUMNS]

    monkeypatch.setattr(rp, "fetch_sp500_tickers", lambda: ["AAPL"])
    monkeypatch.setattr(rp, "refresh_fundamentals_yfinance", lambda tickers: DummyRefreshResult(df, 1, 1, 0, False, []))

    result = run_stock_fundamentals_pipeline(
        cache_path=str(cache_path),
        health_report_path=str(report_path),
        tickers=["AAPL"],
        max_age_days=0.0,
    )

    assert result.data is not None
    assert cache_path.exists()
    assert report_path.exists()
    assert result.wrote_cache is True


def test_pipeline_handles_empty_refresh_without_crashing(tmp_path, monkeypatch):
    cache_path = tmp_path / "fundamentals.parquet"
    report_path = tmp_path / "health.json"

    import data_pipeline.run_pipeline as rp

    empty_df = pd.DataFrame(columns=rp.SCHEMA_COLUMNS)
    monkeypatch.setattr(rp, "fetch_sp500_tickers", lambda: ["ZZZZ"])
    monkeypatch.setattr(
        rp,
        "refresh_fundamentals_yfinance",
        lambda tickers: DummyRefreshResult(empty_df, 1, 0, 1, False, ["failed"]),
    )

    result = run_stock_fundamentals_pipeline(
        cache_path=str(cache_path),
        health_report_path=str(report_path),
        tickers=["ZZZZ"],
        max_age_days=0.0,
    )

    assert result.data is not None
    assert report_path.exists()
    # Cache should not be written when refresh returned no rows.
    assert result.wrote_cache is False
