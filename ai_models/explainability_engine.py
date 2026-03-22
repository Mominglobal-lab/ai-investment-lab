from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai_models.quality_score_model import DEFAULT_WEIGHTS


def _asof_date_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        s = -s
    return s.rank(pct=True, method="average").fillna(0.5)


def _safe_json_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _normalize_tickers(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    out = out.where(~out.isin(["", "NAN", "NONE"]), pd.NA)
    return out


def _component_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    f = feature_df.copy()
    if "Ticker" not in f.columns:
        f = f.reset_index().rename(columns={"index": "Ticker"})
    if "Ticker" not in f.columns:
        raise ValueError("feature_df must include Ticker column or index")

    comp = pd.DataFrame(index=f.index)
    comp["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    def _col(name: str) -> pd.Series:
        if name in f.columns:
            return f[name]
        return pd.Series(np.nan, index=f.index)

    comp["Revenue_Growth_YoY_Pct"] = _percentile_rank(_col("Revenue_Growth_YoY_Pct"), higher_is_better=True)
    comp["EBITDA_Margin"] = _percentile_rank(_col("EBITDA_Margin"), higher_is_better=True)
    comp["ROE"] = _percentile_rank(_col("ROE"), higher_is_better=True)
    comp["FreeCashFlow_Margin"] = _percentile_rank(_col("FreeCashFlow_Margin"), higher_is_better=True)
    comp["Volatility_63D_stability"] = _percentile_rank(_col("Volatility_63D"), higher_is_better=False)
    comp["Drawdown_252D_stability"] = _percentile_rank(_col("Drawdown_252D").abs(), higher_is_better=False)
    return comp


def build_quality_explanations(feature_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    if quality_df is None or quality_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "FeatureAsOfDate",
                "QualityScore",
                "QualityTier",
                "TopPositiveDrivers",
                "TopNegativeDrivers",
                "ContributionJSON",
            ]
        )

    q = quality_df.copy()
    q["Ticker"] = _normalize_tickers(q["Ticker"])
    q = q.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    if q.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "FeatureAsOfDate",
                "QualityScore",
                "QualityTier",
                "TopPositiveDrivers",
                "TopNegativeDrivers",
                "ContributionJSON",
            ]
        )
    comp = _component_table(feature_df)
    comp["Ticker"] = _normalize_tickers(comp["Ticker"])
    comp = comp.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    merged = q.merge(comp, on="Ticker", how="left")

    weights = dict(DEFAULT_WEIGHTS)
    total_w = float(sum(weights.values()))
    weights = {k: float(v / total_w) for k, v in weights.items()}

    feature_names = [
        "Revenue_Growth_YoY_Pct",
        "EBITDA_Margin",
        "ROE",
        "FreeCashFlow_Margin",
        "Volatility_63D_stability",
        "Drawdown_252D_stability",
    ]
    medians = {f: float(pd.to_numeric(merged[f], errors="coerce").median()) for f in feature_names}

    outputs: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        raw_signed: dict[str, float] = {}
        for f in feature_names:
            comp_v = float(row.get(f, 0.5)) if pd.notna(row.get(f)) else 0.5
            delta = comp_v - medians[f]
            raw_signed[f] = float(delta * weights[f])

        norm = float(sum(abs(v) for v in raw_signed.values()))
        if norm > 0:
            signed = {k: float(v / norm) for k, v in raw_signed.items()}
        else:
            signed = {k: 0.0 for k in raw_signed}

        pos = [k for k, v in sorted(signed.items(), key=lambda kv: kv[1], reverse=True) if v > 0][:3]
        neg = [k for k, v in sorted(signed.items(), key=lambda kv: kv[1]) if v < 0][:3]

        outputs.append(
            {
                "Ticker": row["Ticker"],
                "FeatureAsOfDate": _asof_date_iso(),
                "QualityScore": _safe_json_float(row.get("QualityScore")),
                "QualityTier": str(row.get("QualityTier")) if pd.notna(row.get("QualityTier")) else "Unknown",
                "TopPositiveDrivers": ", ".join(pos) if pos else "None",
                "TopNegativeDrivers": ", ".join(neg) if neg else "None",
                "ContributionJSON": json.dumps({k: _safe_json_float(v) for k, v in signed.items()}, sort_keys=True, allow_nan=False),
            }
        )

    return pd.DataFrame(outputs)
