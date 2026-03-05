from __future__ import annotations

import numpy as np
import pandas as pd

from ai_models.regime_detection_model import run_regime_detection_model


def test_regime_model_risk_off_classification(tmp_path):
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    # Build benchmark with a volatility jump in the second half.
    rng1 = np.sin(np.linspace(0, 6, 40)) * 0.002
    rng2 = np.sin(np.linspace(0, 10, 40)) * 0.03
    rets = np.concatenate([rng1, rng2])
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
            "10Y": np.full(len(dates), 4.0),
            "2Y": np.full(len(dates), 4.3),  # Inverted
            "3M": np.full(len(dates), 4.1),
        }
    )
    ty_path = tmp_path / "treasury.parquet"
    ty.to_parquet(ty_path, index=False)

    out = run_regime_detection_model(
        prices_path=str(px_path),
        treasury_path=str(ty_path),
        benchmark_ticker="SPY",
        persistence_days=2,
    )
    assert not out.empty
    assert {"Date", "RegimeLabel", "ConfidenceScore"}.issubset(out.columns)
    assert out["RegimeLabel"].iloc[-1] in {"Risk Off", "Neutral", "Risk On"}

