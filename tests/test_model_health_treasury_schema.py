from __future__ import annotations

import json

import pandas as pd

import data_pipeline.run_pipeline as rp
from ai_models.feature_builder import FeatureBuildResult


def test_model_pipeline_health_accepts_flexible_treasury_schema(tmp_path, monkeypatch):
    treasury_path = tmp_path / "treasury.parquet"
    pd.DataFrame(
        [
            {"AsOfDate": "2025-01-02", "DGS10": 4.0, "DGS2": 3.8},
            {"AsOfDate": "2025-01-03", "DGS10": 4.1, "DGS2": 3.9},
        ]
    ).to_parquet(treasury_path, index=False)

    monkeypatch.setattr(
        rp,
        "build_feature_table",
        lambda **kwargs: FeatureBuildResult(
            features=pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0}]).set_index("Ticker"),
            warnings=[],
            input_coverage={},
        ),
    )
    monkeypatch.setattr(rp, "run_quality_score_model", lambda features: pd.DataFrame([{"Ticker": "AAPL", "QualityScore": 80.0}]))
    monkeypatch.setattr(rp, "run_regime_detection_model", lambda **kwargs: pd.DataFrame([{"Date": "2025-01-02", "RegimeLabel": "Risk On"}]))
    monkeypatch.setattr(rp, "run_systemic_risk_detector", lambda **kwargs: pd.DataFrame([{"Date": "2025-01-02", "RiskLevel": "Low", "RiskScore": 20.0}]))
    monkeypatch.setattr(rp, "save_parquet_atomic", lambda df, path: None)
    monkeypatch.setattr(rp, "get_cache_status", lambda path, max_age_days, required_columns=None: type("S", (), {"exists": True, "schema_ok": True, "is_fresh": True})())

    health_path = tmp_path / "model_health.json"
    rp.run_decision_models_pipeline(
        treasury_path=str(treasury_path),
        model_health_path=str(health_path),
        quality_out_path=str(tmp_path / "quality.parquet"),
        regime_out_path=str(tmp_path / "regime.parquet"),
        risk_out_path=str(tmp_path / "risk.parquet"),
        model_registry_path=str(tmp_path / "registry.json"),
    )

    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["input_cache_coverage"]["treasury_yields_cache"]["exists"] is True
    assert health["input_cache_coverage"]["treasury_yields_cache"]["schema_ok"] is True
    assert health["input_cache_coverage"]["treasury_yields_cache"]["row_count"] == 2
