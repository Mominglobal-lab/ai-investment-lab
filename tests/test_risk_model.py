from __future__ import annotations

import numpy as np
import pandas as pd

from ai_models.risk_detector import run_systemic_risk_detector


def test_risk_score_rises_with_volatility_spike(tmp_path):
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    calm = np.full(200, 0.0003)
    spike = np.sin(np.linspace(0, 20, 60)) * 0.04
    rets = np.concatenate([calm, spike])
    prices = 100 * np.cumprod(1 + rets)

    px = pd.DataFrame(
        {
            "Ticker": ["SPY"] * len(dates),
            "Date": dates,
            "AdjClose": prices,
            "Close": prices,
            "Volume": 1_000_000,
        }
    )
    px_path = tmp_path / "prices.parquet"
    px.to_parquet(px_path, index=False)

    ty = pd.DataFrame(
        {
            "Date": dates,
            "10Y": np.linspace(4.0, 4.2, len(dates)),
            "2Y": np.linspace(4.1, 4.3, len(dates)),
            "3M": np.linspace(3.9, 4.0, len(dates)),
        }
    )
    ty_path = tmp_path / "treasury.parquet"
    ty.to_parquet(ty_path, index=False)

    out = run_systemic_risk_detector(
        prices_path=str(px_path),
        treasury_path=str(ty_path),
        benchmark_ticker="SPY",
    )
    assert not out.empty
    early = out["RiskScore"].head(80).mean()
    late = out["RiskScore"].tail(40).mean()
    assert late > early

