from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ai_models.evidence_builder import build_regime_evidence, build_risk_evidence


def test_regime_evidence_rule_trigger_and_json_keys():
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    # synthetic price series
    rets = np.concatenate([np.full(40, 0.0002), np.sin(np.linspace(0, 12, 40)) * 0.02])
    px = 100 * np.cumprod(1 + rets)
    prices_df = pd.DataFrame(
        {"Ticker": ["SPY"] * len(dates), "Date": dates, "AdjClose": px, "Close": px, "Volume": 1_000_000}
    )
    treasury_df = pd.DataFrame({"Date": dates, "10Y": 4.0, "2Y": 4.3, "3M": 4.1})
    regime_df = pd.DataFrame({"Date": dates, "RegimeLabel": ["Risk Off"] * len(dates), "ConfidenceScore": [0.8] * len(dates)})

    out = build_regime_evidence(prices_df=prices_df, treasury_df=treasury_df, regime_df=regime_df)
    assert not out.empty
    assert out["RuleTriggered"].iloc[-1] in {"inversion_and_volatility_rising", "neutral_default", "steepening_and_volatility_falling"}
    ev = json.loads(out["EvidencePointsJSON"].iloc[-1])
    for k in ["YC_Slope_10Y_2Y", "YC_Slope_10Y_3M", "YC_Inversion", "BenchmarkVol_63d", "BenchmarkTrend_63d"]:
        assert k in ev
    assert isinstance(out["ShortExplanation"].iloc[-1], str) and out["ShortExplanation"].iloc[-1]


def test_risk_evidence_volatility_spike_driver_present():
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    risk_score = np.concatenate([np.linspace(10, 20, 30), np.linspace(20, 80, 30)])
    risk_df = pd.DataFrame({"Date": dates, "RiskScore": risk_score, "RiskLevel": ["Moderate"] * len(dates)})
    prices_df = pd.DataFrame({"Ticker": ["SPY"] * len(dates), "Date": dates, "AdjClose": np.linspace(100, 110, len(dates)), "Close": np.linspace(100, 110, len(dates)), "Volume": 1_000_000})
    treasury_df = pd.DataFrame({"Date": dates, "10Y": 4.0, "2Y": 4.1, "3M": 4.0})

    out = build_risk_evidence(prices_df=prices_df, treasury_df=treasury_df, risk_df=risk_df)
    assert not out.empty
    assert "TopRiskDrivers" in out.columns
    last_drivers = str(out["TopRiskDrivers"].iloc[-1]).lower()
    assert "volatility" in last_drivers or "drawdown" in last_drivers or "rate" in last_drivers
    ev = json.loads(out["EvidencePointsJSON"].iloc[-1])
    required = ["VolatilityExpansion", "RapidDrawdown", "CorrelationSpike", "YieldCurveInversion", "RateShock"]
    for k in required:
        assert k in ev


def test_regime_evidence_uses_selected_benchmark():
    dates = pd.date_range("2025-01-01", periods=90, freq="B")
    prices_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "Ticker": ["SPY"] * len(dates),
                    "Date": dates,
                    "AdjClose": [100.0] * len(dates),
                    "Close": [100.0] * len(dates),
                    "Volume": [1_000_000] * len(dates),
                }
            ),
            pd.DataFrame(
                {
                    "Ticker": ["QQQ"] * len(dates),
                    "Date": dates,
                    "AdjClose": np.linspace(100, 140, len(dates)),
                    "Close": np.linspace(100, 140, len(dates)),
                    "Volume": [1_000_000] * len(dates),
                }
            ),
        ],
        ignore_index=True,
    )
    treasury_df = pd.DataFrame({"Date": dates, "10Y": 4.0, "2Y": 3.7, "3M": 3.8})
    regime_df = pd.DataFrame({"Date": dates, "RegimeLabel": ["Risk On"] * len(dates), "ConfidenceScore": [0.8] * len(dates)})

    out = build_regime_evidence(
        prices_df=prices_df,
        treasury_df=treasury_df,
        regime_df=regime_df,
        benchmark_ticker="QQQ",
    )

    ev = json.loads(out["EvidencePointsJSON"].iloc[-1])
    assert ev["BenchmarkTrend_63d"] is not None
    assert ev["BenchmarkTrend_63d"] > 0


def test_regime_evidence_does_not_overstate_unconfirmed_conditions():
    dates = pd.date_range("2025-01-01", periods=90, freq="B")
    prices_df = pd.DataFrame(
        {
            "Ticker": ["SPY"] * len(dates),
            "Date": dates,
            "AdjClose": [100.0] * len(dates),
            "Close": [100.0] * len(dates),
            "Volume": [1_000_000] * len(dates),
        }
    )
    treasury_df = pd.DataFrame({"Date": dates, "10Y": 4.0, "2Y": 3.5, "3M": 3.7})
    regime_df = pd.DataFrame({"Date": dates, "RegimeLabel": ["Risk Off"] * len(dates), "ConfidenceScore": [0.8] * len(dates)})

    out = build_regime_evidence(prices_df=prices_df, treasury_df=treasury_df, regime_df=regime_df)

    assert out["RuleTriggered"].iloc[-1] == "risk_off_label_without_full_confirmation"
    assert "not both confirmed" in out["ShortExplanation"].iloc[-1]


def test_risk_evidence_json_sanitizes_non_finite_values():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    risk_df = pd.DataFrame({"Date": dates, "RiskScore": [10.0, float("inf"), 30.0], "RiskLevel": ["Low", "Low", "Moderate"]})

    out = build_risk_evidence(prices_df=pd.DataFrame(), treasury_df=None, risk_df=risk_df)

    assert not out.empty
    text = out["EvidencePointsJSON"].iloc[-1]
    assert "Infinity" not in text
    assert "NaN" not in text
    parsed = json.loads(text)
    assert isinstance(parsed, dict)

