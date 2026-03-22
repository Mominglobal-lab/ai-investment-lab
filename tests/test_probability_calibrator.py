from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_models.probability_calibrator import build_regime_probabilities


def test_probabilities_sum_to_one_and_confidence_mapping():
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RegimeLabel": ["Risk On"] * 6,
            "ConfidenceScore": [0.30, 0.40, 0.50, 0.60, 0.75, 0.95],
        }
    )
    out = build_regime_probabilities(df, window=3)
    s = out[["P_RiskOn", "P_Neutral", "P_RiskOff"]].sum(axis=1)
    assert np.allclose(s.values, 1.0, atol=1e-8)
    assert out["P_RiskOn"].iloc[-1] > out["P_RiskOn"].iloc[1]


def test_regime_stability_behaves_for_stable_and_flipping_series():
    dates = pd.date_range("2025-01-01", periods=25, freq="B")
    stable = pd.DataFrame({"Date": dates, "RegimeLabel": ["Neutral"] * 25, "ConfidenceScore": [0.6] * 25})
    flip_labels = ["Risk On" if i % 2 == 0 else "Risk Off" for i in range(25)]
    flipping = pd.DataFrame({"Date": dates, "RegimeLabel": flip_labels, "ConfidenceScore": [0.6] * 25})

    s1 = build_regime_probabilities(stable, window=20)
    s2 = build_regime_probabilities(flipping, window=20)
    assert s1["RegimeStability_20d"].iloc[-1] > s2["RegimeStability_20d"].iloc[-1]


def test_missing_regime_labels_default_to_neutral():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RegimeLabel": [None, "", "nan"],
            "ConfidenceScore": [0.7, 0.7, 0.7],
        }
    )

    out = build_regime_probabilities(df, window=2)

    assert set(out["RegimeLabel"]) == {"Neutral"}
    assert (out["P_Neutral"] > out["P_RiskOn"]).all()
    assert (out["P_Neutral"] > out["P_RiskOff"]).all()


def test_initial_regime_stability_starts_at_one_for_constant_series():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RegimeLabel": ["Neutral", "Neutral", "Neutral"],
            "ConfidenceScore": [0.6, 0.6, 0.6],
        }
    )

    out = build_regime_probabilities(df, window=3)

    assert float(out["RegimeStability_20d"].iloc[0]) == 1.0

