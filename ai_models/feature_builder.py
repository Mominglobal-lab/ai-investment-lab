from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FUNDAMENTALS_PATH = "data/fundamentals_cache.parquet"
PRICES_PATH = "data/prices_cache.parquet"
TREASURY_PATH = "data/treasury_yields_cache.parquet"


@dataclass(frozen=True)
class FeatureBuildResult:
    features: pd.DataFrame
    warnings: list[str]
    input_coverage: dict[str, Any]


def _safe_read_parquet(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_fundamentals(fundamentals_path: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    df = _safe_read_parquet(fundamentals_path)
    if df is None or df.empty:
        fallback_paths = [
            "data/fundamentals_cache_sp500.parquet",
            "data/fundamentals_cache_nasdaq100.parquet",
        ]
        frames: list[pd.DataFrame] = []
        for fp in fallback_paths:
            f = _safe_read_parquet(fp)
            if f is not None and not f.empty:
                frames.append(f.copy())
        if frames:
            df = pd.concat(frames, axis=0, ignore_index=True)
            warnings.append("Primary fundamentals cache missing; used segmented fundamentals caches.")
        else:
            return pd.DataFrame(), ["Fundamentals cache not found or empty."]

    out = df.copy()
    if "Ticker" not in out.columns:
        return pd.DataFrame(), ["Fundamentals cache missing Ticker column."]

    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Revenue_Growth_YoY_Pct", "EBITDA_Margin", "ROE", "FreeCashFlow_Margin"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Conservative fallback: if explicit FCF margin is absent, keep NaN and impute later in model.
    cols = ["Ticker", "Revenue_Growth_YoY_Pct", "EBITDA_Margin", "ROE", "FreeCashFlow_Margin"]
    out = out[cols].dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last")
    return out, warnings


def _build_price_features(prices_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = prices_df.copy()
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
    px["AdjClose"] = pd.to_numeric(px["AdjClose"], errors="coerce")
    px = px.dropna(subset=["Ticker", "Date", "AdjClose"]).sort_values(["Ticker", "Date"])

    features: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    for ticker, g in px.groupby("Ticker"):
        s = g.set_index("Date")["AdjClose"].sort_index()
        r1 = s.pct_change()
        feat = {
            "Ticker": ticker,
            "Momentum_12M": float(s.pct_change(252).iloc[-1]) if len(s) > 252 else np.nan,
            "Volatility_63D": float(r1.rolling(63).std().iloc[-1] * np.sqrt(252)) if len(r1) > 63 else np.nan,
            "Drawdown_252D": np.nan,
            "Return_21D": float(s.pct_change(21).iloc[-1]) if len(s) > 21 else np.nan,
        }
        if len(s) >= 30:
            roll_max = s.rolling(252, min_periods=30).max()
            dd = (s / roll_max) - 1.0
            feat["Drawdown_252D"] = float(dd.iloc[-1]) if not dd.empty else np.nan
        features.append(feat)

        if ticker == "SPY":
            spy_vol = r1.rolling(63).std() * np.sqrt(252)
            spy_trend = s.pct_change(252)
            macro_rows = [
                {
                    "Date": idx,
                    "Benchmark_Volatility": float(spy_vol.loc[idx]) if pd.notna(spy_vol.loc[idx]) else np.nan,
                    "Benchmark_Trend": float(spy_trend.loc[idx]) if pd.notna(spy_trend.loc[idx]) else np.nan,
                }
                for idx in s.index
            ]

    out = pd.DataFrame(features)
    macro = pd.DataFrame(macro_rows) if macro_rows else pd.DataFrame(columns=["Date", "Benchmark_Volatility", "Benchmark_Trend"])
    return out, macro


def _detect_col(columns: list[str], candidates: list[str]) -> str | None:
    canon = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in canon:
            return canon[key]
    return None


def _build_yield_features(treasury_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    if treasury_df is None or treasury_df.empty:
        return pd.DataFrame(columns=["Date", "YC_10Y_2Y", "YC_10Y_3M", "YC_Inversion", "YC_Volatility"]), [
            "Treasury yield cache unavailable; macro yield features set to neutral defaults."
        ]

    df = treasury_df.copy()
    date_col = _detect_col(list(df.columns), ["Date", "AsOfDate", "Timestamp"])
    if date_col is None:
        return pd.DataFrame(columns=["Date", "YC_10Y_2Y", "YC_10Y_3M", "YC_Inversion", "YC_Volatility"]), [
            "Treasury cache missing date column."
        ]

    y10_col = _detect_col(list(df.columns), ["10Y", "DGS10", "Yield10Y", "UST10Y"])
    y2_col = _detect_col(list(df.columns), ["2Y", "DGS2", "Yield2Y", "UST2Y"])
    y3m_col = _detect_col(list(df.columns), ["3M", "DGS3MO", "Yield3M", "UST3M"])
    if y10_col is None or (y2_col is None and y3m_col is None):
        return pd.DataFrame(columns=["Date", "YC_10Y_2Y", "YC_10Y_3M", "YC_Inversion", "YC_Volatility"]), [
            "Treasury cache missing expected yield columns."
        ]

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    y10 = pd.to_numeric(df[y10_col], errors="coerce")
    y2 = pd.to_numeric(df[y2_col], errors="coerce") if y2_col else np.nan
    y3m = pd.to_numeric(df[y3m_col], errors="coerce") if y3m_col else np.nan

    out["YC_10Y_2Y"] = y10 - y2 if y2_col else np.nan
    out["YC_10Y_3M"] = y10 - y3m if y3m_col else np.nan
    slope = out["YC_10Y_2Y"].fillna(out["YC_10Y_3M"])
    out["YC_Inversion"] = (slope < 0).astype(float)
    out["YC_Volatility"] = slope.rolling(21, min_periods=5).std()
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out, []


def build_feature_table(
    *,
    fundamentals_path: str = FUNDAMENTALS_PATH,
    prices_path: str = PRICES_PATH,
    treasury_path: str = TREASURY_PATH,
) -> FeatureBuildResult:
    warnings: list[str] = []
    fundamentals, fund_warn = _load_fundamentals(fundamentals_path)
    warnings.extend(fund_warn)

    prices = _safe_read_parquet(prices_path)
    if prices is None or prices.empty:
        raise FileNotFoundError(f"Price cache missing or empty: {prices_path}")
    if not {"Ticker", "Date", "AdjClose"}.issubset(set(prices.columns)):
        raise ValueError("Price cache missing required columns: Ticker, Date, AdjClose")

    price_features, macro_series = _build_price_features(prices)
    latest_macro_vol = float(macro_series["Benchmark_Volatility"].dropna().iloc[-1]) if not macro_series.empty and not macro_series["Benchmark_Volatility"].dropna().empty else np.nan
    latest_macro_trend = float(macro_series["Benchmark_Trend"].dropna().iloc[-1]) if not macro_series.empty and not macro_series["Benchmark_Trend"].dropna().empty else np.nan

    treasury = _safe_read_parquet(treasury_path)
    yield_series, y_warn = _build_yield_features(treasury)
    warnings.extend(y_warn)
    latest_yield = yield_series.tail(1)
    yc_10_2 = float(latest_yield["YC_10Y_2Y"].iloc[0]) if not latest_yield.empty else np.nan
    yc_10_3m = float(latest_yield["YC_10Y_3M"].iloc[0]) if not latest_yield.empty else np.nan
    yc_inv = float(latest_yield["YC_Inversion"].iloc[0]) if not latest_yield.empty else 0.0
    yc_vol = float(latest_yield["YC_Volatility"].iloc[0]) if not latest_yield.empty else np.nan

    merged = pd.merge(
        price_features,
        fundamentals,
        on="Ticker",
        how="left",
    )
    merged["Benchmark_Volatility"] = latest_macro_vol
    merged["Benchmark_Trend"] = latest_macro_trend
    merged["YC_10Y_2Y"] = yc_10_2
    merged["YC_10Y_3M"] = yc_10_3m
    merged["YC_Inversion"] = yc_inv
    merged["YC_Volatility"] = yc_vol

    merged = merged.drop_duplicates(subset=["Ticker"], keep="last").set_index("Ticker").sort_index()

    coverage = {
        "fundamentals_rows": int(len(fundamentals)),
        "prices_rows": int(len(prices)),
        "feature_rows": int(len(merged)),
        "treasury_rows": int(len(treasury) if treasury is not None else 0),
        "missing_counts": {c: int(merged[c].isna().sum()) for c in merged.columns},
    }
    return FeatureBuildResult(features=merged, warnings=warnings, input_coverage=coverage)

