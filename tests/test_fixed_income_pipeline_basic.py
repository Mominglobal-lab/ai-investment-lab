from __future__ import annotations

import pandas as pd

from data_pipeline.run_pipeline import run_fixed_income_pipeline


class DummyRefreshResult:
    def __init__(self, df: pd.DataFrame, requested: int, success: int, failed: int, rate_limited: bool, errors: list[str]):
        self.data = df
        self.requested_count = requested
        self.success_count = success
        self.failure_count = failed
        self.rate_limited = rate_limited
        self.errors_sample = errors


def test_fixed_income_pipeline_writes_cache_and_report(tmp_path, monkeypatch):
    cache_path = tmp_path / "fixed_income.parquet"
    report_path = tmp_path / "health.json"

    import data_pipeline.run_pipeline as rp

    df = pd.DataFrame(
        [
            {
                "Symbol": "IEF",
                "Name": "iShares 7-10 Year Treasury Bond ETF",
                "Universe": "US Treasuries",
                "Type": "Treasury ETF",
                "Price": 95.0,
                "Yield_Pct": 4.1,
                "Duration_Years": 7.3,
                "Maturity_Bucket": "7-10Y",
                "Expense_Ratio_Pct": 0.15,
                "AUM": 29_000_000_000.0,
            }
        ]
    )
    for col in rp.FI_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[rp.FI_SCHEMA_COLUMNS]

    monkeypatch.setattr(rp, "fetch_fixed_income_universe_instruments", lambda universe: [{"Symbol": "IEF"}])
    monkeypatch.setattr(rp, "refresh_fixed_income_yfinance", lambda instruments: DummyRefreshResult(df, 1, 1, 0, False, []))

    result = run_fixed_income_pipeline(
        cache_path=str(cache_path),
        health_report_path=str(report_path),
        universe="US Treasuries",
        max_age_days=0.0,
    )
    assert result.data is not None
    assert result.wrote_cache is True
    assert cache_path.exists()
    assert report_path.exists()


def test_fixed_income_pipeline_handles_empty_refresh_without_crashing(tmp_path, monkeypatch):
    cache_path = tmp_path / "fixed_income.parquet"
    report_path = tmp_path / "health.json"

    import data_pipeline.run_pipeline as rp

    empty_df = pd.DataFrame(columns=rp.FI_SCHEMA_COLUMNS)
    monkeypatch.setattr(rp, "fetch_fixed_income_universe_instruments", lambda universe: [{"Symbol": "ZZZ"}])
    monkeypatch.setattr(
        rp,
        "refresh_fixed_income_yfinance",
        lambda instruments: DummyRefreshResult(empty_df, 1, 0, 1, False, ["failed"]),
    )

    result = run_fixed_income_pipeline(
        cache_path=str(cache_path),
        health_report_path=str(report_path),
        universe="US Treasuries",
        max_age_days=0.0,
    )
    assert result.data is not None
    assert result.wrote_cache is False
    assert report_path.exists()
