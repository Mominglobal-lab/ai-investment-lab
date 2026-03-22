from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.cache_manager import write_json_report


def test_write_json_report_sanitizes_non_finite_floats(tmp_path):
    path = Path(tmp_path) / "report.json"
    report = {
        "ok": True,
        "finite": 1.25,
        "nan_value": float("nan"),
        "inf_value": float("inf"),
        "nested": {"neg_inf": float("-inf")},
        "items": [1.0, float("nan"), False],
    }

    write_json_report(report, str(path))

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    assert "NaN" not in text
    assert "Infinity" not in text
    assert data["ok"] is True
    assert data["finite"] == 1.25
    assert data["nan_value"] is None
    assert data["inf_value"] is None
    assert data["nested"]["neg_inf"] is None
    assert data["items"] == [1.0, None, False]
