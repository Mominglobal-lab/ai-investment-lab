from __future__ import annotations

import json

import pandas as pd

import data_pipeline.run_pipeline as rp
from ai_models.feature_builder import FeatureBuildResult


def test_registry_upserts_preserve_existing_layers_without_duplicates(tmp_path, monkeypatch):
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "generated_at": "2025-01-01T00:00:00+00:00",
                "models": [
                    {"model_name": "quality_score_model", "timestamp": "old"},
                    {"model_name": "explainability_layer", "timestamp": "old"},
                ],
            }
        ),
        encoding="utf-8",
    )

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
    monkeypatch.setattr(rp, "get_cache_status", lambda path, max_age_days, required_columns=None: type("S", (), {"exists": False, "schema_ok": False, "is_fresh": False})())
    monkeypatch.setattr(rp, "read_parquet_safe", lambda path: (pd.DataFrame(), None))

    rp.run_decision_models_pipeline(
        model_registry_path=str(registry_path),
        quality_out_path=str(tmp_path / "quality.parquet"),
        regime_out_path=str(tmp_path / "regime.parquet"),
        risk_out_path=str(tmp_path / "risk.parquet"),
        model_health_path=str(tmp_path / "health.json"),
    )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    names = [m["model_name"] for m in registry["models"]]

    assert names.count("quality_score_model") == 1
    assert names.count("explainability_layer") == 1
    assert "regime_detection_model" in names
    assert "risk_detector" in names
