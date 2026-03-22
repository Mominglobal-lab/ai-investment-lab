from __future__ import annotations

import pandas as pd

from data_pipeline.data_fetcher import refresh_prices_yfinance


def test_refresh_prices_yfinance_respects_requested_lookback(monkeypatch):
    requested_periods: list[str] = []

    class DummyTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, period: str, interval: str, auto_adjust: bool):
            requested_periods.append(period)
            return pd.DataFrame(
                {
                    "Date": pd.date_range("2025-01-01", periods=3, freq="B"),
                    "Adj Close": [100.0, 101.0, 102.0],
                    "Close": [100.0, 101.0, 102.0],
                    "Volume": [1_000_000, 1_000_000, 1_000_000],
                }
            ).set_index("Date")

    class DummyYF:
        @staticmethod
        def Ticker(symbol: str):
            return DummyTicker(symbol)

    monkeypatch.setitem(__import__("sys").modules, "yfinance", DummyYF)

    out = refresh_prices_yfinance(["AAPL"], lookback_years=1)

    assert not out.data.empty
    assert requested_periods == ["1y"]
