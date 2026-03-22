from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai_models.quality_score_model import DEFAULT_WEIGHTS


def _asof_date_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _tier_from_score(score: float) -> str:
    if score >= 67:
        return "Strong"
    if score >= 34:
        return "Neutral"
    return "Weak"


def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        s = -s
    return s.rank(pct=True, method="average").fillna(0.5)


def _normalize_tickers(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    out = out.where(~out.isin(["", "NAN", "NONE"]), pd.NA)
    return out


def _normalize_quality_tier(value: object, default: str) -> str:
    txt = str(value).strip() if pd.notna(value) else ""
    if txt in {"", "nan", "None"}:
        return default
    return txt


def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "Ticker" not in x.columns:
        x = x.reset_index().rename(columns={"index": "Ticker"})
    if "Ticker" not in x.columns:
        raise ValueError("feature_df must include Ticker column or index")

    def col(name: str) -> pd.Series:
        if name in x.columns:
            return x[name]
        return pd.Series(np.nan, index=x.index)

    comp = pd.DataFrame(index=x.index)
    comp["Ticker"] = x["Ticker"].astype(str).str.upper().str.strip()
    comp["Revenue_Growth_YoY_Pct"] = _percentile_rank(col("Revenue_Growth_YoY_Pct"), True)
    comp["EBITDA_Margin"] = _percentile_rank(col("EBITDA_Margin"), True)
    comp["ROE"] = _percentile_rank(col("ROE"), True)
    comp["FreeCashFlow_Margin"] = _percentile_rank(col("FreeCashFlow_Margin"), True)
    comp["Volatility_63D_stability"] = _percentile_rank(col("Volatility_63D"), False)
    comp["Drawdown_252D_stability"] = _percentile_rank(col("Drawdown_252D").abs(), False)

    weights = dict(DEFAULT_WEIGHTS)
    w_sum = float(sum(weights.values()))
    weights = {k: float(v / w_sum) for k, v in weights.items()}
    score = sum(comp[c] * weights.get(c, 0.0) for c in weights.keys()) * 100.0
    out = pd.DataFrame({"Ticker": comp["Ticker"], "Score": score.astype(float)})
    out["Tier"] = out["Score"].map(_tier_from_score)
    return out


def build_quality_uncertainty(
    feature_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    if quality_df is None or quality_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "FeatureAsOfDate",
                "ScoreMean",
                "ScoreP10",
                "ScoreP50",
                "ScoreP90",
                "TierMostLikely",
                "TierStability",
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
                "ScoreMean",
                "ScoreP10",
                "ScoreP50",
                "ScoreP90",
                "TierMostLikely",
                "TierStability",
            ]
        )

    f = feature_df.copy()
    if "Ticker" not in f.columns:
        f = f.reset_index().rename(columns={"index": "Ticker"})
    f["Ticker"] = _normalize_tickers(f["Ticker"])
    f = f.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    if f.empty:
        out = q.copy()
        out["FeatureAsOfDate"] = _asof_date_iso()
        out["ScoreMean"] = pd.to_numeric(out.get("QualityScore"), errors="coerce")
        out["ScoreP10"] = out["ScoreMean"]
        out["ScoreP50"] = out["ScoreMean"]
        out["ScoreP90"] = out["ScoreMean"]
        out["TierMostLikely"] = [
            _normalize_quality_tier(v, default="Unknown")
            for v in out.get("QualityTier", pd.Series(["Unknown"] * len(out)))
        ]
        out["TierStability"] = 1.0
        return out[
            ["Ticker", "FeatureAsOfDate", "ScoreMean", "ScoreP10", "ScoreP50", "ScoreP90", "TierMostLikely", "TierStability"]
        ]

    rng = np.random.default_rng(int(seed))
    n = len(f)
    by_ticker_scores: dict[str, list[float]] = {t: [] for t in q["Ticker"].tolist()}
    by_ticker_tiers: dict[str, list[str]] = {t: [] for t in q["Ticker"].tolist()}

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sample = f.iloc[idx].copy().reset_index(drop=True)
        scored = _score_frame(sample)
        grouped = scored.groupby("Ticker", as_index=False)["Score"].mean()
        grouped["Tier"] = grouped["Score"].map(_tier_from_score)
        gmap = grouped.set_index("Ticker")
        for t in by_ticker_scores.keys():
            if t in gmap.index:
                s = float(gmap.loc[t, "Score"])
                tier = str(gmap.loc[t, "Tier"])
                by_ticker_scores[t].append(s)
                by_ticker_tiers[t].append(tier)

    rows: list[dict[str, object]] = []
    for _, r in q.iterrows():
        t = str(r["Ticker"])
        vals = by_ticker_scores.get(t, [])
        tiers = by_ticker_tiers.get(t, [])
        if not vals:
            point = float(pd.to_numeric(r.get("QualityScore"), errors="coerce")) if pd.notna(r.get("QualityScore")) else np.nan
            vals = [point] if pd.notna(point) else [50.0]
            tier_guess = _normalize_quality_tier(r.get("QualityTier"), default=_tier_from_score(vals[0]))
            tiers = [tier_guess]

        arr = np.array(vals, dtype=float)
        p10 = float(np.nanpercentile(arr, 10))
        p50 = float(np.nanpercentile(arr, 50))
        p90 = float(np.nanpercentile(arr, 90))
        mean = float(np.nanmean(arr))
        vc = pd.Series(tiers).value_counts()
        tier_mode = str(vc.index[0]) if not vc.empty else _tier_from_score(mean)
        stability = float((pd.Series(tiers) == tier_mode).mean()) if tiers else 0.0

        rows.append(
            {
                "Ticker": t,
                "FeatureAsOfDate": _asof_date_iso(),
                "ScoreMean": mean,
                "ScoreP10": p10,
                "ScoreP50": p50,
                "ScoreP90": p90,
                "TierMostLikely": tier_mode,
                "TierStability": stability,
            }
        )
    return pd.DataFrame(rows)


def build_risk_uncertainty(risk_df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    empty_cols = [
        "Date",
        "RiskScore",
        "RiskP10",
        "RiskP50",
        "RiskP90",
        "RiskLevelMostLikely",
        "RiskLevelStability",
    ]
    if risk_df is None or risk_df.empty:
        return pd.DataFrame(columns=empty_cols)

    r = risk_df.copy()
    r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
    r["RiskScore"] = pd.to_numeric(r["RiskScore"], errors="coerce")
    r.loc[~np.isfinite(r["RiskScore"]), "RiskScore"] = np.nan
    r = r.dropna(subset=["Date", "RiskScore"]).sort_values("Date").reset_index(drop=True)
    if r.empty:
        return pd.DataFrame(columns=empty_cols)

    deltas = r["RiskScore"].diff()

    def lvl(v: float) -> str:
        if v <= 35:
            return "Low"
        if v <= 65:
            return "Moderate"
        return "Elevated"

    out_rows: list[dict[str, object]] = []
    for i, row in r.iterrows():
        hist = deltas.iloc[max(1, i - int(window)) : i + 1].dropna()
        if len(hist) < 10:
            p10 = p50 = p90 = float(row["RiskScore"])
        else:
            q10 = float(np.nanpercentile(hist.values, 10))
            q50 = float(np.nanpercentile(hist.values, 50))
            q90 = float(np.nanpercentile(hist.values, 90))
            p10 = float(np.clip(row["RiskScore"] + q10, 0, 100))
            p50 = float(np.clip(row["RiskScore"] + q50, 0, 100))
            p90 = float(np.clip(row["RiskScore"] + q90, 0, 100))

        levels = [lvl(p10), lvl(p50), lvl(p90)]
        vc = pd.Series(levels).value_counts()
        most = str(vc.index[0]) if not vc.empty else lvl(float(row["RiskScore"]))
        stab = float((pd.Series(levels) == most).mean()) if levels else 0.0

        out_rows.append(
            {
                "Date": row["Date"],
                "RiskScore": float(row["RiskScore"]),
                "RiskP10": p10,
                "RiskP50": p50,
                "RiskP90": p90,
                "RiskLevelMostLikely": most,
                "RiskLevelStability": stab,
            }
        )
    return pd.DataFrame(out_rows)

