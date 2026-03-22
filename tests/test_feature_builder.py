from __future__ import annotations

import pandas as pd

from ai_models.feature_builder import build_feature_table


def test_build_feature_table_uses_selected_benchmark(tmp_path):
    fundamentals_path = tmp_path / "fundamentals.parquet"
    prices_path = tmp_path / "prices.parquet"
    treasury_path = tmp_path / "treasury.parquet"

    pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "Revenue_Growth_YoY_Pct": 10.0,
                "EBITDA_Margin": 0.30,
                "ROE": 0.25,
                "FreeCashFlow_Margin": 0.15,
            }
        ]
    ).to_parquet(fundamentals_path, index=False)

    dates = pd.bdate_range("2025-01-01", periods=320)
    price_rows: list[dict[str, object]] = []
    for i, dt in enumerate(dates):
        price_rows.append({"Ticker": "SPY", "Date": dt, "AdjClose": 100.0, "Close": 100.0, "Volume": 1_000_000})
        price_rows.append(
            {
                "Ticker": "QQQ",
                "Date": dt,
                "AdjClose": 100.0 + i,
                "Close": 100.0 + i,
                "Volume": 1_000_000,
            }
        )
        price_rows.append(
            {
                "Ticker": "AAPL",
                "Date": dt,
                "AdjClose": 150.0 + (i * 0.5),
                "Close": 150.0 + (i * 0.5),
                "Volume": 1_000_000,
            }
        )
    pd.DataFrame(price_rows).to_parquet(prices_path, index=False)

    pd.DataFrame(
        [
            {"Date": dt, "10Y": 4.0, "2Y": 3.8, "3M": 3.9}
            for dt in dates
        ]
    ).to_parquet(treasury_path, index=False)

    out = build_feature_table(
        fundamentals_path=str(fundamentals_path),
        prices_path=str(prices_path),
        treasury_path=str(treasury_path),
        benchmark_ticker="QQQ",
    )

    features = out.features.reset_index()
    assert not features.empty
    assert "Benchmark_Trend" in features.columns
    assert features["Benchmark_Trend"].notna().all()
    assert float(features["Benchmark_Trend"].iloc[0]) > 0.0
