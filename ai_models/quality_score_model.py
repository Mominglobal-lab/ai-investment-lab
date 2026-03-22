from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "Revenue_Growth_YoY_Pct": 0.20,
    "EBITDA_Margin": 0.20,
    "ROE": 0.20,
    "FreeCashFlow_Margin": 0.20,
    "Volatility_63D_stability": 0.10,
    "Drawdown_252D_stability": 0.10,
}


def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        s = -s
    rank = s.rank(pct=True, method="average")
    # Conservative neutral imputation for missing values.
    return rank.fillna(0.5)


def _tier_from_score(score: float) -> str:
    if score >= 67:
        return "Strong"
    if score >= 34:
        return "Neutral"
    return "Weak"


def _normalize_tickers(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    out = out.where(~out.isin(["", "NAN", "NONE"]), pd.NA)
    return out


def run_quality_score_model(
    features: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame(columns=["Ticker", "QualityScore", "QualityTier", "Explanation"])

    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total_w = float(sum(w.values()))
    if total_w <= 0:
        raise ValueError("weights must sum to positive value")
    w = {k: float(v / total_w) for k, v in w.items()}

    df = features.copy()
    if "Ticker" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Ticker"})
    if "Ticker" not in df.columns:
        raise ValueError("features must include Ticker column or index")

    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(np.nan, index=df.index)

    components = pd.DataFrame(index=df.index)
    components["Revenue_Growth_YoY_Pct"] = _percentile_rank(_col("Revenue_Growth_YoY_Pct"), higher_is_better=True)
    components["EBITDA_Margin"] = _percentile_rank(_col("EBITDA_Margin"), higher_is_better=True)
    components["ROE"] = _percentile_rank(_col("ROE"), higher_is_better=True)
    components["FreeCashFlow_Margin"] = _percentile_rank(_col("FreeCashFlow_Margin"), higher_is_better=True)
    components["Volatility_63D_stability"] = _percentile_rank(_col("Volatility_63D"), higher_is_better=False)
    components["Drawdown_252D_stability"] = _percentile_rank(_col("Drawdown_252D").abs(), higher_is_better=False)

    weighted = sum(components[c] * w.get(c, 0.0) for c in components.columns)
    score = (weighted * 100.0).clip(0, 100)

    tickers = _normalize_tickers(df["Ticker"])
    valid_mask = tickers.notna()
    components = components.loc[valid_mask].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "Ticker": tickers.loc[valid_mask].reset_index(drop=True),
            "QualityScore": score.loc[valid_mask].astype(float).reset_index(drop=True),
        }
    )
    out["QualityTier"] = out["QualityScore"].map(_tier_from_score)

    labels = {
        "Revenue_Growth_YoY_Pct": "Revenue growth",
        "EBITDA_Margin": "EBITDA margin",
        "ROE": "Return on equity",
        "FreeCashFlow_Margin": "Free cash flow margin",
        "Volatility_63D_stability": "Volatility stability",
        "Drawdown_252D_stability": "Drawdown stability",
    }

    explanations: list[str] = []
    for idx in out.index:
        contrib = {c: float(components.loc[idx, c] * w.get(c, 0.0)) for c in components.columns}
        top3 = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]
        explanations.append(", ".join([labels[k] for k, _ in top3]))
    out["Explanation"] = explanations

    out = out.sort_values("QualityScore", ascending=False).reset_index(drop=True)
    return out[["Ticker", "QualityScore", "QualityTier", "Explanation"]]

