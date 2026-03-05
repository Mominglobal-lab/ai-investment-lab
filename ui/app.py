from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.cache_manager import get_cache_status, read_parquet_safe, save_parquet_atomic, write_json_report
from data_pipeline.data_fetcher import (
    FI_SCHEMA_COLUMNS,
    SCHEMA_COLUMNS,
    fetch_ticker_details,
    fetch_universe_tickers,
    refresh_fundamentals_yfinance,
)
from data_pipeline.run_pipeline import run_fixed_income_pipeline, run_pipeline, run_prices_cache_pipeline
from reports.decision_brief import generate_decision_brief
from simulation.portfolio_simulator import simulate_portfolio

st.set_page_config(page_title="Investment Lab", layout="wide")

STOCK_UNIVERSE_OPTIONS = ["S&P 500", "Nasdaq 100"]
FI_UNIVERSE_OPTIONS = ["US Treasuries", "Bond ETFs"]
MAX_AGE_DAYS = 7
MIN_REFRESH_SUCCESS_RATIO = 0.25
PRICE_SCHEMA_COLUMNS = ["Ticker", "Date", "AdjClose", "Close", "Volume"]
PRICES_CACHE_PATH = "data/prices_cache.parquet"
PRICES_HEALTH_PATH = "data/prices_health_report.json"
QUALITY_CACHE_PATH = "data/quality_scores_cache.parquet"
REGIME_CACHE_PATH = "data/regime_cache.parquet"
RISK_CACHE_PATH = "data/risk_signals_cache.parquet"
QUALITY_EXPLAIN_PATH = "data/quality_explanations_cache.parquet"
REGIME_EVIDENCE_PATH = "data/regime_evidence_cache.parquet"
RISK_EVIDENCE_PATH = "data/risk_evidence_cache.parquet"


def _apply_premium_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #071225;
            --bg-1: #0b1a33;
            --bg-2: #0f2745;
            --surface: #0c1b32;
            --surface-2: #0f223d;
            --border: #1c3a63;
            --text: #e8f1ff;
            --muted: #9db3d6;
            --accent-cyan: #2ec5ff;
            --accent-teal: #27d4a6;
            --accent-orange: #ff9f43;
            --shadow: 0 12px 26px rgba(2, 8, 20, 0.40);
        }

        .stApp {
            background:
                radial-gradient(1200px 700px at 0% 0%, #154b79 0%, transparent 50%),
                radial-gradient(1000px 700px at 100% 0%, #1ea38e 0%, transparent 45%),
                linear-gradient(165deg, var(--bg-0) 10%, var(--bg-1) 50%, var(--bg-2) 100%);
            color: var(--text);
        }

        .block-container {
            max-width: none;
            width: 100%;
            padding-top: 5.2rem;
            padding-bottom: 1rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        [data-testid="stHeader"] {
            background: linear-gradient(90deg, rgba(8, 18, 36, 0.95), rgba(14, 48, 77, 0.92));
            border-bottom: 1px solid #28496f;
            backdrop-filter: blur(6px);
            min-height: 86px;
        }

        [data-testid="stHeader"]::before {
            content: "Intelligent Investment Assistant";
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            color: #e8f1ff;
            font-weight: 800;
            font-size: 3rem;
            letter-spacing: 0.35px;
            text-align: center;
            white-space: nowrap;
            pointer-events: none;
            z-index: 2;
        }

        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] svg,
        [data-testid="stToolbar"] span,
        [data-testid="stToolbar"] div {
            color: #d7e8ff !important;
            fill: #d7e8ff !important;
        }

        h1, h2, h3, h4 {
            color: var(--text);
        }

        .stApp, .stApp p, .stApp label, .stApp span, .stApp small {
            color: var(--text);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        [data-testid="stHorizontalBlock"] > div:has([data-testid="stTextInput"]),
        [data-testid="stHorizontalBlock"] > div:has([data-testid="stSelectbox"]),
        [data-testid="stHorizontalBlock"] > div:has([data-testid="stMultiSelect"]),
        [data-testid="stHorizontalBlock"] > div:has([data-testid="stNumberInput"]),
        [data-testid="stHorizontalBlock"] > div:has([data-testid="stSlider"]),
        [data-testid="stHorizontalBlock"] > div:has([data-testid="stCheckbox"]),
        [data-testid="stHorizontalBlock"] > div:has([data-testid="stButton"]) {
            background: linear-gradient(180deg, rgba(12, 27, 50, 0.93), rgba(12, 27, 50, 0.82));
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px 12px 12px 12px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(6px);
            min-height: 92px;
        }

        [data-testid="stHorizontalBlock"] > div:has([data-testid="stButton"]) {
            display: flex;
            align-items: center;
            justify-content: center;
            padding-top: 8px;
            padding-bottom: 8px;
        }

        [data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(12, 27, 50, 0.93), rgba(12, 27, 50, 0.82));
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--shadow);
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div {
            background: #102746 !important;
            border: 1px solid #244670 !important;
            border-radius: 10px !important;
            color: var(--text) !important;
        }

        [data-testid="stSelectbox"] *[data-baseweb="select"] span,
        [data-testid="stMultiSelect"] *[data-baseweb="select"] span {
            color: var(--text) !important;
        }

        /* BaseWeb popover menu (select/multiselect options) */
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="option"] {
            background: #102746 !important;
            color: #dce9ff !important;
        }

        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background: #17355a !important;
            color: #f2f7ff !important;
        }

        div[data-baseweb="popover"] [role="option"]:hover {
            background: #1a3b63 !important;
            color: #ffffff !important;
        }

        [data-testid="stTextInput"] input::placeholder {
            color: #7f98bd !important;
        }

        [data-testid="stButton"] > button {
            background: linear-gradient(135deg, #1ea8ff, #2fd7b0);
            color: #041224;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            min-height: 42px;
            box-shadow: 0 6px 14px rgba(16, 158, 220, 0.22);
        }

        [data-testid="stButton"] > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        [data-testid="stCheckbox"] label {
            color: #c7daf6 !important;
        }

        [data-testid="stSlider"] * {
            color: #c9ddfb !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #b8c6dc;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: var(--shadow);
            background: #ffffff;
        }

        [data-testid="stDataFrame"] thead th,
        [data-testid="stDataFrame"] [role="columnheader"] {
            font-weight: 900 !important;
            color: #0a0f18 !important;
            background: #f0f4fa !important;
            border-bottom: 1px solid #d5e0ee !important;
        }

        [data-testid="stDataFrame"] [role="columnheader"] *,
        [data-testid="stDataFrame"] [role="columnheader"] div,
        [data-testid="stDataFrame"] [role="columnheader"] span {
            font-weight: 900 !important;
            color: #0a0f18 !important;
        }

        [data-testid="stDataFrame"] td,
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] span,
        [data-testid="stDataFrame"] div {
            color: #060b12 !important;
        }

        [data-testid="stDataFrame"] tbody tr {
            background: #ffffff !important;
        }

        [data-testid="stDataFrame"] tbody tr:nth-child(even) {
            background: #f9fbff !important;
        }

        [data-testid="stDataFrame"] td {
            border-top: 1px solid #e2e8f2 !important;
        }

        [data-testid="stDataFrame"] td:has(span[title="None"]),
        [data-testid="stDataFrame"] td:has(div[title="None"]) {
            color: #7f8fa8 !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 10px;
            margin-bottom: 0.15rem;
        }

        [data-testid="stTabs"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding-top: 0.15rem !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 999px;
            border: 1px solid #2a4d79;
            background: #102746;
            color: #c2d7f7;
            padding: 8px 16px;
            font-weight: 600;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(46, 197, 255, 0.2), rgba(39, 212, 166, 0.2));
            border-color: #41cde8 !important;
            color: #e8f7ff !important;
            box-shadow: 0 8px 20px rgba(46, 197, 255, 0.2);
        }

        .stAlert {
            border-radius: 12px;
            border: 1px solid #27507e;
            background: rgba(16, 39, 70, 0.9);
        }

        .stDivider {
            border-top: 1px solid #28496f;
        }

        .ii-table-wrap {
            max-height: 500px;
            overflow: auto;
            border: 1px solid #b8c6dc;
            border-radius: 14px;
            box-shadow: var(--shadow);
            background: #ffffff;
        }

        .ii-table-wrap table {
            width: 100%;
        }

        .ii-table-wrap thead th {
            color: #05070b !important;
            font-weight: 900 !important;
            background: #edf3fb !important;
        }

        .ii-table-wrap tbody td {
            color: #04070d !important;
            font-weight: 500 !important;
            background: transparent !important;
        }

        .ii-insights {
            background: linear-gradient(180deg, rgba(12, 27, 50, 0.96), rgba(12, 27, 50, 0.90));
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 0.5rem;
        }

        .ii-insights h4 {
            margin: 0 0 6px 0;
            color: #e8f1ff;
            font-size: 1.05rem;
        }

        .ii-insights ul {
            margin: 0;
            padding-left: 18px;
        }

        .ii-insights li {
            margin: 1px 0;
            line-height: 1.2;
            color: #dce9ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_premium_theme()


@dataclass(frozen=True)
class StockUpdateResult:
    data: pd.DataFrame
    requested_count: int
    success_count: int
    failure_count: int
    rate_limited: bool
    errors_sample: list[str]
    wrote_cache: bool
    reason: str


def _stock_paths(universe: str) -> tuple[str, str]:
    key = (universe or "").strip().lower().replace("&", "and").replace(" ", "").replace("-", "")
    if key in {"sandp500", "sp500"}:
        return "data/fundamentals_cache_sp500.parquet", "data/fundamentals_health_report_sp500.json"
    if key in {"nasdaq100", "ndx"}:
        return "data/fundamentals_cache_nasdaq100.parquet", "data/fundamentals_health_report_nasdaq100.json"
    raise ValueError(f"Unsupported stock universe: {universe}")


def _fi_paths(universe: str) -> tuple[str, str]:
    key = (universe or "").strip().lower().replace(" ", "").replace("-", "")
    if key in {"ustreasuries", "treasury", "treasuries"}:
        return "data/fixed_income_cache_treasury.parquet", "data/fixed_income_health_treasury.json"
    if key in {"bondetfs", "bondetf", "etf"}:
        return "data/fixed_income_cache_bond_etf.parquet", "data/fixed_income_health_bond_etf.json"
    raise ValueError(f"Unsupported fixed-income universe: {universe}")


def _load_fundamentals_union() -> pd.DataFrame:
    paths = [
        "data/fundamentals_cache.parquet",
        "data/fundamentals_cache_sp500.parquet",
        "data/fundamentals_cache_nasdaq100.parquet",
    ]
    frames: list[pd.DataFrame] = []
    for p in paths:
        df, _err = read_parquet_safe(p)
        if df is None or df.empty:
            continue
        if "Ticker" not in df.columns:
            continue
        keep_cols = [c for c in ["Ticker", "MarketCap"] if c in df.columns]
        frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame(columns=["Ticker", "MarketCap"])
    out = pd.concat(frames, axis=0, ignore_index=True)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["MarketCap"] = pd.to_numeric(out.get("MarketCap"), errors="coerce")
    out = out.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last")
    return out


def _parse_ticker_input(text: str) -> list[str]:
    raw = [x.strip().upper() for x in str(text or "").split(",")]
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _build_holdings(
    tickers: list[str],
    weighting_mode: str,
    fundamentals_df: pd.DataFrame,
    manual_weights_text: str,
) -> tuple[list[tuple[str, float]], list[str]]:
    warnings: list[str] = []
    if not tickers:
        return [], ["No tickers provided"]

    if weighting_mode == "Equal weight":
        w = 1.0 / len(tickers)
        return [(t, w) for t in tickers], warnings

    if weighting_mode == "Market cap weight":
        if fundamentals_df.empty:
            w = 1.0 / len(tickers)
            warnings.append("Fundamentals cache unavailable for market-cap weighting. Using equal weights.")
            return [(t, w) for t in tickers], warnings
        mcap_map = fundamentals_df.set_index("Ticker")["MarketCap"].to_dict()
        vals = [float(mcap_map.get(t, 0.0) or 0.0) for t in tickers]
        total = float(sum([v for v in vals if v > 0]))
        if total <= 0:
            w = 1.0 / len(tickers)
            warnings.append("Selected tickers missing market-cap values. Using equal weights.")
            return [(t, w) for t in tickers], warnings
        return [(t, float(max(v, 0.0) / total)) for t, v in zip(tickers, vals)], warnings

    tokens = [x.strip() for x in str(manual_weights_text or "").split(",") if x.strip()]
    if len(tokens) != len(tickers):
        return [], [f"Manual weights count ({len(tokens)}) must match ticker count ({len(tickers)})"]
    weights: list[float] = []
    for tok in tokens:
        try:
            weights.append(float(tok))
        except Exception:
            return [], [f"Invalid manual weight: {tok}"]
    return list(zip(tickers, weights)), warnings


def _load_latest_model_signals() -> tuple[str, str]:
    regime_label = "Unknown"
    risk_level = "Unknown"

    regime_df, _err = read_parquet_safe(REGIME_CACHE_PATH)
    if regime_df is not None and not regime_df.empty and "RegimeLabel" in regime_df.columns:
        regime_label = str(regime_df.iloc[-1]["RegimeLabel"])

    risk_df, _err = read_parquet_safe(RISK_CACHE_PATH)
    if risk_df is not None and not risk_df.empty and "RiskLevel" in risk_df.columns:
        risk_level = str(risk_df.iloc[-1]["RiskLevel"])

    return regime_label, risk_level


def _show_signal_banner() -> None:
    regime_label, risk_level = _load_latest_model_signals()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Market Regime:** {regime_label}")
    with c2:
        st.markdown(f"**Systemic Risk:** {risk_level}")


def _ensure_stock_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[SCHEMA_COLUMNS]
    out["PEG_Ratio"] = (pd.to_numeric(out["PE_Ratio"], errors="coerce") / pd.to_numeric(out["Earnings_Growth_Pct"], errors="coerce")).where(
        pd.to_numeric(out["Earnings_Growth_Pct"], errors="coerce") > 0
    )
    out["Rule_of_40"] = pd.to_numeric(out["Revenue_Growth_YoY_Pct"], errors="coerce") + (
        pd.to_numeric(out["EBITDA_Margin"], errors="coerce") * 100.0
    )
    return out


def _styled_table(df: pd.DataFrame):
    show = df.copy()
    numeric_cols = show.select_dtypes(include=["number"]).columns.tolist()
    fmt = {c: "{:,.2f}" for c in numeric_cols}

    styler = show.style.format(fmt, na_rep="-").set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#f0f4fa"),
                    ("color", "#0a0f18"),
                    ("font-weight", "900"),
                    ("border-bottom", "1px solid #d5e0ee"),
                    ("border-right", "1px solid #d5e0ee"),
                    ("padding", "10px 8px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("color", "#04070d"),
                    ("font-weight", "500"),
                    ("border-color", "#d9e4f2"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #c4d4e8"),
                    ("font-size", "14px"),
                ],
            },
        ]
    )
    styler = styler.set_properties(**{"color": "#04070d", "font-weight": "500"})
    styler = styler.apply(
        lambda row: ["background-color: #f4f8ff" if (row.name % 2 == 1) else "background-color: #ffffff" for _ in row],
        axis=1,
    )
    return styler


def _render_styled_table(df: pd.DataFrame) -> None:
    styler = _styled_table(df).hide(axis="index")
    st.markdown(f'<div class="ii-table-wrap">{styler.to_html()}</div>', unsafe_allow_html=True)


def _update_stock_cache(cache_path: str, health_report_path: str, universe: str, include_metadata: bool) -> StockUpdateResult:
    tickers = fetch_universe_tickers(universe)
    refresh = refresh_fundamentals_yfinance(tickers, include_metadata=include_metadata)
    df = _ensure_stock_schema(refresh.data)

    reason = "cache updated"
    wrote_cache = False
    if df.empty:
        reason = "no rows returned from refresh"
    elif refresh.requested_count > 0 and (refresh.success_count / refresh.requested_count) < MIN_REFRESH_SUCCESS_RATIO:
        reason = f"success ratio {(refresh.success_count / refresh.requested_count):.1%} below threshold {MIN_REFRESH_SUCCESS_RATIO:.0%}"
    else:
        save_parquet_atomic(df, cache_path)
        wrote_cache = True

    report = {
        "run_timestamp": pd.Timestamp.utcnow().isoformat(),
        "universe": universe,
        "requested_count": int(refresh.requested_count),
        "success_count": int(refresh.success_count),
        "failure_count": int(refresh.failure_count),
        "rate_limited": bool(refresh.rate_limited),
        "wrote_cache": bool(wrote_cache),
        "reason": str(reason),
        "errors_sample": list(refresh.errors_sample or []),
    }
    write_json_report(report, health_report_path)

    return StockUpdateResult(
        data=df,
        requested_count=refresh.requested_count,
        success_count=refresh.success_count,
        failure_count=refresh.failure_count,
        rate_limited=refresh.rate_limited,
        errors_sample=refresh.errors_sample,
        wrote_cache=wrote_cache,
        reason=reason,
    )


def _show_stock_tab() -> None:
    st.markdown("## Stock Screener")
    status_text = ""
    telemetry_text = ""
    left, mid, right = st.columns([2.3, 1.0, 1.4], vertical_alignment="bottom")
    with left:
        st.caption("Universe")
        universe = st.selectbox(
            "Universe",
            STOCK_UNIVERSE_OPTIONS,
            index=0,
            key="stock_universe",
            label_visibility="collapsed",
        )
    with mid:
        st.caption(" ")
        include_metadata = st.checkbox("Enrich metadata (slower)", value=False, key="stock_enrich")
    with right:
        st.caption(" ")
        force_refresh = st.button("Refresh Stock Data", use_container_width=True, key="stock_refresh")

    cache_path, health_report_path = _stock_paths(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=SCHEMA_COLUMNS)
    status_text = (
        ("No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days")
        + f" | {'Fresh' if status.is_fresh else 'Stale'} | {'Schema OK' if status.schema_ok else 'Schema mismatch'}"
    )

    stale_df, _err = read_parquet_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
    if stale_df is not None:
        stale_df = _ensure_stock_schema(stale_df)

    needs_refresh = force_refresh or stale_df is None or (not status.is_fresh) or (not status.schema_ok)
    if needs_refresh:
        with st.spinner("Refreshing stock data..."):
            try:
                update = _update_stock_cache(cache_path, health_report_path, universe, include_metadata)
                if update.wrote_cache:
                    df_raw = update.data
                elif stale_df is not None and not stale_df.empty:
                    df_raw = stale_df
                    st.warning(f"Refresh incomplete ({update.reason}). Using stale cache.")
                else:
                    df_raw = update.data
                    st.warning(f"Refresh incomplete ({update.reason}).")
                telemetry_text = f"Updated {update.success_count}/{update.requested_count} tickers"
            except Exception as e:
                if stale_df is not None and not stale_df.empty:
                    df_raw = stale_df
                    st.warning(f"Refresh failed ({e}). Using stale cache.")
                else:
                    st.error(f"Refresh failed and no usable cache is available: {e}")
                    return
    else:
        df_raw = stale_df
        telemetry_text = "Using fresh stock cache"

    if df_raw is None or df_raw.empty:
        st.warning("No stock data available.")
        return

    df = df_raw.copy()
    qdf, _qerr = read_parquet_safe(QUALITY_CACHE_PATH)
    if qdf is not None and not qdf.empty and "Ticker" in qdf.columns:
        qtmp = qdf.copy()
        qtmp["Ticker"] = qtmp["Ticker"].astype(str).str.upper().str.strip()
        if "QualityScore" in qtmp.columns:
            qtmp["QualityScore"] = pd.to_numeric(qtmp["QualityScore"], errors="coerce")
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df = df.merge(qtmp[["Ticker", "QualityScore", "QualityTier"]], on="Ticker", how="left")
    else:
        df["QualityScore"] = pd.NA
        df["QualityTier"] = pd.NA

    f1, f2, f3, f4 = st.columns([2.0, 2.0, 1.6, 1.6], vertical_alignment="bottom")
    with f1:
        query = st.text_input("Search ticker/company", value="", key="stock_query").strip().lower()
    with f2:
        sector_options = sorted([s for s in df["Sector"].dropna().unique().tolist()])
        sectors = st.multiselect("Sectors", options=sector_options, key="stock_sectors")
    with f3:
        min_rule40 = st.number_input(
            "Rule of 40 min",
            min_value=-50.0,
            max_value=80.0,
            value=-50.0,
            step=1.0,
            key="stock_rule40",
        )
    def _reset_stock_filters() -> None:
        defaults = {
            "stock_query": "",
            "stock_sectors": [],
            "stock_rule40": -50.0,
            "stock_ebitda_min": 0.0,
            "stock_roe_min": 0.0,
            "stock_rev_growth_min": -50.0,
            "stock_earnings_growth_min": -50.0,
            "stock_pe_min": 0.0,
            "stock_peg_min": 0.0,
            "stock_mcap_min_b": 0.0,
            "stock_mcap_max_b": 10000.0,
            "stock_require_complete": False,
            "stock_quality_min": 0.0,
            "stock_sort_by": "MarketCap",
        }
        for k, v in defaults.items():
            st.session_state[k] = v

    with f4:
        st.button("Reset Filters", use_container_width=True, key="stock_reset", on_click=_reset_stock_filters)

    with st.expander("Advanced Screening Filters", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            ebitda_min = st.number_input("EBITDA margin min", min_value=0.0, max_value=0.8, value=0.0, step=0.01, key="stock_ebitda_min")
        with s2:
            roe_min = st.number_input("ROE min", min_value=0.0, max_value=0.8, value=0.0, step=0.01, key="stock_roe_min")
        with s3:
            revenue_growth_min = st.number_input(
                "Revenue growth YoY min (Pct)",
                min_value=-50.0,
                max_value=200.0,
                value=-50.0,
                step=1.0,
                key="stock_rev_growth_min",
            )
        with s4:
            earnings_growth_min = st.number_input(
                "Earnings growth min (Pct)",
                min_value=-50.0,
                max_value=200.0,
                value=-50.0,
                step=1.0,
                key="stock_earnings_growth_min",
            )

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            pe_min = st.number_input("P/E ratio min", min_value=0.0, max_value=200.0, value=0.0, step=1.0, key="stock_pe_min")
        with v2:
            peg_min = st.number_input("PEG ratio min", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="stock_peg_min")
        with v3:
            mcap_min_b = st.number_input("Market cap min (B)", min_value=0.0, max_value=10000.0, value=0.0, step=1.0, key="stock_mcap_min_b")
        with v4:
            mcap_max_b = st.number_input(
                "Market cap max (B)",
                min_value=0.0,
                max_value=10000.0,
                value=10000.0,
                step=10.0,
                key="stock_mcap_max_b",
            )

        require_complete = st.checkbox("Require complete data", value=False, key="stock_require_complete")
        q1, q2 = st.columns(2)
        with q1:
            quality_min = st.slider(
                "Quality score min",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="stock_quality_min",
            )
        with q2:
            sort_by = st.selectbox(
                "Sort by",
                ["MarketCap", "QualityScore"],
                index=0,
                key="stock_sort_by",
            )

    if query:
        df = df[
            df["Ticker"].astype(str).str.lower().str.contains(query, na=False)
            | df["Company"].astype(str).str.lower().str.contains(query, na=False)
        ]
    if sectors:
        df = df[df["Sector"].isin(sectors)]
    allow_missing = not require_complete

    if ebitda_min > 0:
        mask = df["EBITDA_Margin"] >= ebitda_min
        df = df[mask | df["EBITDA_Margin"].isna()] if allow_missing else df[mask]
    if roe_min > 0:
        mask = df["ROE"] >= roe_min
        df = df[mask | df["ROE"].isna()] if allow_missing else df[mask]
    if revenue_growth_min > -50:
        mask = df["Revenue_Growth_YoY_Pct"] >= revenue_growth_min
        df = df[mask | df["Revenue_Growth_YoY_Pct"].isna()] if allow_missing else df[mask]
    if earnings_growth_min > -50:
        mask = df["Earnings_Growth_Pct"] >= earnings_growth_min
        df = df[mask | df["Earnings_Growth_Pct"].isna()] if allow_missing else df[mask]
    if min_rule40 > -50:
        mask = df["Rule_of_40"] >= min_rule40
        df = df[mask | df["Rule_of_40"].isna()] if allow_missing else df[mask]
    if pe_min > 0:
        mask = df["PE_Ratio"] >= pe_min
        df = df[mask | df["PE_Ratio"].isna()] if allow_missing else df[mask]
    if peg_min > 0:
        mask = df["PEG_Ratio"] >= peg_min
        df = df[mask | df["PEG_Ratio"].isna()] if allow_missing else df[mask]
    if mcap_min_b > 0:
        mask = df["MarketCap"] >= (mcap_min_b * 1e9)
        df = df[mask | df["MarketCap"].isna()] if allow_missing else df[mask]
    if mcap_max_b < 10000:
        mask = df["MarketCap"] <= (mcap_max_b * 1e9)
        df = df[mask | df["MarketCap"].isna()] if allow_missing else df[mask]
    if quality_min > 0:
        mask = pd.to_numeric(df["QualityScore"], errors="coerce") >= quality_min
        df = df[mask | df["QualityScore"].isna()] if allow_missing else df[mask]

    if require_complete:
        df = df.dropna(
            subset=[
                "EBITDA_Margin",
                "ROE",
                "Revenue_Growth_YoY_Pct",
                "Earnings_Growth_Pct",
                "PE_Ratio",
                "PEG_Ratio",
                "Rule_of_40",
                "MarketCap",
            ]
        )
    sort_col = "QualityScore" if sort_by == "QualityScore" else "MarketCap"
    df = df.sort_values(by=sort_col, ascending=False, na_position="last")

    df_show = df.copy()
    df_show["MarketCap (B)"] = (pd.to_numeric(df_show["MarketCap"], errors="coerce") / 1e9).round(2)
    df_show["EBITDA Margin (Pct)"] = (pd.to_numeric(df_show["EBITDA_Margin"], errors="coerce") * 100).round(2)
    df_show["ROE (Pct)"] = (pd.to_numeric(df_show["ROE"], errors="coerce") * 100).round(2)
    df_show = df_show[
        [
            "Ticker",
            "Company",
            "Sector",
            "QualityScore",
            "QualityTier",
            "Close",
            "MarketCap (B)",
            "Revenue_Growth_YoY_Pct",
            "Earnings_Growth_Pct",
            "PE_Ratio",
            "PEG_Ratio",
            "Rule_of_40",
            "EBITDA Margin (Pct)",
            "ROE (Pct)",
        ]
    ]
    st.write(f"Matches: {len(df_show):,}")
    _render_styled_table(df_show)

    ticker = st.selectbox("Ticker details", options=[""] + df["Ticker"].astype(str).tolist(), index=0, key="stock_detail_ticker")
    if ticker:
        try:
            d = fetch_ticker_details(ticker)
            st.markdown(f"**{d.get('Company') or ticker} ({ticker})**")
            st.write(f"Sector: {d.get('Sector')}")
            st.write(f"Close: {d.get('Close')}")
            st.write(f"P/E: {d.get('PE_Ratio')}")
            st.write(f"PEG: {d.get('PEG_Ratio')}")
            st.write(f"Rule of 40: {d.get('Rule_of_40')}")
        except Exception as e:
            st.caption(f"Live detail unavailable: {e}")

    st.divider()
    if status_text:
        st.caption(status_text)
    if telemetry_text:
        st.caption(telemetry_text)


def _show_fixed_income_tab() -> None:
    st.markdown("## Bond & Treasury Screener")
    status_text = ""
    telemetry_text = ""
    left, right = st.columns([2.3, 1.4], vertical_alignment="bottom")
    with left:
        st.caption("Universe")
        universe = st.selectbox(
            "Universe",
            FI_UNIVERSE_OPTIONS,
            index=0,
            key="fi_universe",
            label_visibility="collapsed",
        )
    with right:
        st.caption(" ")
        force_refresh = st.button("Refresh Fixed-Income Data", use_container_width=True, key="fi_refresh")

    cache_path, health_report_path = _fi_paths(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=FI_SCHEMA_COLUMNS)
    status_text = (
        ("No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days")
        + f" | {'Fresh' if status.is_fresh else 'Stale'} | {'Schema OK' if status.schema_ok else 'Schema mismatch'}"
    )

    stale_df, _err = read_parquet_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
    needs_refresh = force_refresh or stale_df is None or (not status.is_fresh) or (not status.schema_ok)
    if needs_refresh:
        with st.spinner("Refreshing fixed-income data..."):
            result = run_fixed_income_pipeline(
                cache_path=cache_path,
                health_report_path=health_report_path,
                universe=universe,
                max_age_days=MAX_AGE_DAYS,
            )
            df_raw = result.data
            if (not result.wrote_cache) and stale_df is not None and not stale_df.empty:
                df_raw = stale_df
                st.warning(f"Refresh incomplete ({result.reason}). Using stale cache.")
            telemetry_text = result.reason
    else:
        df_raw = stale_df
        telemetry_text = "Using fresh fixed-income cache"

    if df_raw is None or df_raw.empty:
        st.warning("No fixed-income data available.")
        return

    df = df_raw.copy()
    for col in ["Price", "Yield_Pct", "Duration_Years", "Expense_Ratio_Pct", "AUM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        q = st.text_input("Search symbol/name", value="", key="fi_query").strip().lower()
    with c2:
        buckets = sorted([b for b in df["Maturity_Bucket"].dropna().astype(str).unique().tolist()])
        selected_buckets = st.multiselect("Maturity bucket", options=buckets, key="fi_buckets")
    with c3:
        min_yield = st.slider("Yield min (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1, key="fi_yield_min")
    with c4:
        max_duration = st.slider("Duration max (yrs)", min_value=0.0, max_value=30.0, value=30.0, step=0.1, key="fi_dur_max")

    if q:
        df = df[
            df["Symbol"].astype(str).str.lower().str.contains(q, na=False)
            | df["Name"].astype(str).str.lower().str.contains(q, na=False)
        ]
    if selected_buckets:
        df = df[df["Maturity_Bucket"].astype(str).isin(selected_buckets)]
    df = df[df["Yield_Pct"].fillna(-1) >= min_yield]
    df = df[df["Duration_Years"].fillna(999) <= max_duration]
    df = df.sort_values(by="Yield_Pct", ascending=False)

    show = df.copy()
    show["AUM (B)"] = (show["AUM"] / 1e9).round(2)
    show = show[
        [
            "Symbol",
            "Name",
            "Universe",
            "Type",
            "Price",
            "Yield_Pct",
            "Duration_Years",
            "Maturity_Bucket",
            "Expense_Ratio_Pct",
            "AUM (B)",
        ]
    ]
    st.write(f"Matches: {len(show):,}")
    _render_styled_table(show)

    detail_symbol = st.selectbox("Instrument details", options=[""] + df["Symbol"].astype(str).tolist(), index=0, key="fi_detail")
    if detail_symbol:
        row = df[df["Symbol"] == detail_symbol].head(1).iloc[0]
        st.markdown(f"**{row.get('Name')} ({detail_symbol})**")
        st.write(f"Yield (%): {row.get('Yield_Pct')}")
        st.write(f"Duration (yrs): {row.get('Duration_Years')}")
        st.write(f"Expense ratio (%): {row.get('Expense_Ratio_Pct')}")
        st.write(f"AUM: {row.get('AUM')}")
        if pd.notna(row.get("Duration_Years")):
            shock = -float(row["Duration_Years"])
            st.write(f"Estimated price impact for +100 bps move: {shock:.2f}%")

    st.divider()
    if status_text:
        st.caption(status_text)
    if telemetry_text:
        st.caption(telemetry_text)


def _show_portfolio_simulator_tab() -> None:
    title_col, act1, act2, act3 = st.columns([3.2, 1.0, 1.0, 1.0], vertical_alignment="center")
    with title_col:
        st.markdown("## Portfolio Decision Simulator")
    with act1:
        refresh_clicked = st.button("Refresh Data", key="sim_refresh_data_btn", use_container_width=True)
    with act2:
        run_clicked = st.button("Run Simulation", key="sim_run_btn", use_container_width=True)
    with act3:
        export_clicked = st.button("Export Decision Brief", key="sim_export_brief", use_container_width=True)

    price_status = get_cache_status(PRICES_CACHE_PATH, MAX_AGE_DAYS, required_columns=PRICE_SCHEMA_COLUMNS)
    if (not price_status.exists) or (not price_status.schema_ok):
        st.warning("Price cache not found. Please run pipeline refresh.")
        if refresh_clicked:
            with st.spinner("Refreshing pipeline and rebuilding prices cache..."):
                try:
                    run_pipeline(
                        build_prices_cache=True,
                        max_age_days=MAX_AGE_DAYS,
                        benchmark_ticker=str(st.session_state.get("sim_benchmark", "SPY") or "SPY").strip().upper(),
                    )
                    st.success("Prices cache refresh completed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")
        return

    fundamentals = _load_fundamentals_union()
    default_tickers = "AAPL, MSFT, NVDA, GOOGL"

    r1c1, r1c2, r1c3 = st.columns([2.4, 1.2, 1.2], vertical_alignment="bottom")
    with r1c1:
        ticker_text = st.text_input("Tickers (comma-separated)", value=default_tickers, key="sim_tickers")
    with r1c2:
        weighting_mode = st.selectbox(
            "Weighting",
            ["Equal weight", "Market cap weight", "Manual weights"],
            index=0,
            key="sim_weighting_mode",
        )
    with r1c3:
        benchmark = st.text_input("Benchmark", value="SPY", key="sim_benchmark").strip().upper()

    manual_weights_text = ""
    if weighting_mode == "Manual weights":
        manual_weights_text = st.text_input(
            "Manual weights (comma-separated, same order as tickers)",
            value="0.25, 0.25, 0.25, 0.25",
            key="sim_manual_weights",
        )

    if refresh_clicked:
        with st.spinner("Refreshing prices cache..."):
            try:
                refresh_result = run_prices_cache_pipeline(
                    prices_cache_path=PRICES_CACHE_PATH,
                    health_report_path=PRICES_HEALTH_PATH,
                    benchmark_ticker=benchmark or "SPY",
                    max_age_days=0.0,
                )
                st.success(
                    "Prices cache refreshed. "
                    f"Requested: {refresh_result.requested_count}, "
                    f"Success: {refresh_result.success_count}, "
                    f"Failed: {refresh_result.failure_count}."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5, vertical_alignment="bottom")
    with r2c1:
        lookback_years = st.selectbox("Lookback period", [1, 3, 5, 10], index=2, key="sim_lookback")
    with r2c2:
        rebalance_label = st.selectbox("Rebalance", ["None", "Monthly"], index=0, key="sim_rebalance")
    with r2c3:
        mode_label = st.selectbox("Simulation mode", ["Historical", "Monte Carlo"], index=0, key="sim_mode")
    with r2c4:
        initial_capital = float(
            st.number_input(
                "Starting capital ($)",
                min_value=100.0,
                max_value=100000000.0,
                value=10000.0,
                step=1000.0,
                key="sim_initial_capital",
            )
        )
    with r2c5:
        strict_mode = st.checkbox("Strict missing-data mode", value=False, key="sim_strict")

    mc_paths = 1000
    horizon_days = 252
    if mode_label == "Monte Carlo":
        m1, m2 = st.columns(2, vertical_alignment="bottom")
        with m1:
            mc_paths = int(st.number_input("Number of simulations", min_value=100, max_value=20000, value=1000, step=100, key="sim_mc_paths"))
        with m2:
            horizon_years = float(st.number_input("Horizon years", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="sim_horizon_years"))
            horizon_days = int(round(horizon_years * 252))

    tickers = _parse_ticker_input(ticker_text)
    holdings, build_warnings = _build_holdings(
        tickers=tickers,
        weighting_mode=weighting_mode,
        fundamentals_df=fundamentals,
        manual_weights_text=manual_weights_text,
    )
    for w in build_warnings:
        st.warning(w)

    if run_clicked:
        if not holdings:
            st.error("Cannot run simulation: invalid holdings setup.")
        else:
            try:
                result = simulate_portfolio(
                    holdings=holdings,
                    lookback_years=int(lookback_years),
                    rebalance_rule="monthly" if rebalance_label == "Monthly" else "none",
                    benchmark=benchmark or "SPY",
                    mode="monte_carlo" if mode_label == "Monte Carlo" else "historical",
                    monte_carlo_paths=int(mc_paths),
                    horizon_days=int(horizon_days),
                    strict=bool(strict_mode),
                    initial_capital=float(initial_capital),
                )
                result.setdefault("portfolio", {})
                result["portfolio"]["weighting_mode"] = weighting_mode
                st.session_state["portfolio_sim_result"] = result
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    result = st.session_state.get("portfolio_sim_result")
    if not result:
        st.caption("Configure inputs and click Run Simulation.")
        return

    if export_clicked:
        try:
            artifact = generate_decision_brief(
                simulation_result=result,
                output_dir="data/run_artifacts",
                format="html",
                title="Portfolio Decision Brief",
            )
            st.success(
                "Decision Brief exported. "
                f"HTML: {artifact.get('html_path')} | JSON: {artifact.get('json_path')}"
            )
        except Exception as e:
            st.error(f"Decision Brief export failed: {e}")

    ts = result.get("timeseries", {})
    dates = pd.to_datetime(ts.get("dates", []), errors="coerce")
    pvals = pd.Series(ts.get("portfolio_value", []), index=dates, name="Portfolio")
    dds = pd.Series(ts.get("drawdown", []), index=dates, name="Drawdown")
    bvals_raw = ts.get("benchmark_value")
    bvals = pd.Series(bvals_raw, index=dates, name="Benchmark") if bvals_raw is not None else None

    sim_start = pd.to_datetime((result.get("metadata") or {}).get("period_start"), errors="coerce")
    regime_df, _rerr = read_parquet_safe(REGIME_CACHE_PATH)
    if regime_df is not None and not regime_df.empty and {"Date", "RegimeLabel"}.issubset(set(regime_df.columns)) and pd.notna(sim_start):
        rg = regime_df.copy()
        rg["Date"] = pd.to_datetime(rg["Date"], errors="coerce")
        rg = rg.dropna(subset=["Date"]).sort_values("Date")
        at_start = rg[rg["Date"] <= sim_start]
        if at_start.empty:
            label = str(rg.iloc[0]["RegimeLabel"])
        else:
            label = str(at_start.iloc[-1]["RegimeLabel"])
        st.caption(f"Regime at simulation start ({sim_start.strftime('%Y-%m-%d')}): {label}")

    insights = result.get("decision_insights", [])
    if not insights:
        st.caption("No insights available.")
    else:
        insight_items = "".join(f"<li>{line}</li>" for line in insights)
        st.markdown(
            f'<div class="ii-insights"><h4>Decision Insights</h4><ul>{insight_items}</ul></div>',
            unsafe_allow_html=True,
        )

    show_risk_overlay = st.checkbox(
        "Show Risk Overlay (adds a line showing calmer vs riskier periods)",
        value=False,
        key="sim_show_risk_overlay",
    )

    c1, c2 = st.columns(2)
    with c1:
        end_portfolio = float(pvals.dropna().iloc[-1]) if not pvals.dropna().empty else 0.0
        start_capital = float((result.get("metadata") or {}).get("initial_capital", 10000.0))
        if bvals is not None and not bvals.dropna().empty:
            end_benchmark = float(bvals.dropna().iloc[-1])
            growth_title = (
                f"#### Growth of \\${start_capital:,.0f} "
                f"(Portfolio \\${end_portfolio:,.0f} vs. Benchmark \\${end_benchmark:,.0f})"
            )
        else:
            growth_title = (
                f"#### Growth of \\${start_capital:,.0f} "
                f"(Portfolio \\${end_portfolio:,.0f} vs. Benchmark N/A)"
            )
        st.markdown(growth_title)
        growth_df = pd.DataFrame({"Date": pd.to_datetime(pvals.index, errors="coerce"), "Portfolio": pvals.values}).dropna()
        if bvals is not None:
            btmp = pd.DataFrame({"Date": pd.to_datetime(bvals.index, errors="coerce"), "Benchmark": bvals.values}).dropna()
            growth_df = growth_df.merge(btmp, on="Date", how="left")
        growth_long = growth_df.melt(id_vars=["Date"], var_name="Series", value_name="Value").dropna()
        risk_overlay = None
        if show_risk_overlay:
            risk_df, _risk_err = read_parquet_safe(RISK_CACHE_PATH)
            if risk_df is not None and not risk_df.empty and {"Date", "RiskScore"}.issubset(set(risk_df.columns)):
                rtmp = risk_df.copy()
                rtmp["Date"] = pd.to_datetime(rtmp["Date"], errors="coerce")
                rtmp["RiskScore"] = pd.to_numeric(rtmp["RiskScore"], errors="coerce")
                rtmp = rtmp.dropna(subset=["Date", "RiskScore"]).sort_values("Date")
                if not rtmp.empty:
                    risk_overlay = pd.merge_asof(
                        growth_df[["Date"]].sort_values("Date"),
                        rtmp[["Date", "RiskScore"]],
                        on="Date",
                        direction="backward",
                    ).dropna()
        if not growth_long.empty:
            growth_lines = (
                alt.Chart(growth_long)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title=""),
                    y=alt.Y("Value:Q", title=""),
                    color=alt.Color("Series:N", legend=alt.Legend(title="")),
                )
            )
            if risk_overlay is not None and not risk_overlay.empty:
                risk_line = (
                    alt.Chart(risk_overlay)
                    .mark_line(color="#ff9f43", strokeDash=[4, 4])
                    .encode(
                        x=alt.X("Date:T", title=""),
                        y=alt.Y(
                            "RiskScore:Q",
                            title="Risk Score",
                            axis=alt.Axis(orient="right"),
                            scale=alt.Scale(domain=[0, 100]),
                        ),
                    )
                )
                growth_chart = alt.layer(growth_lines, risk_line).resolve_scale(y="independent").properties(height=320).interactive()
            else:
                growth_chart = growth_lines.properties(height=320).interactive()
            st.altair_chart(growth_chart, use_container_width=True)
        else:
            st.caption("No growth data available.")
    with c2:
        st.markdown("#### Drawdown")
        draw_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(dds.index, errors="coerce"),
                "DrawdownPct": (dds * 100.0).round(0),
            }
        ).dropna()
        if not draw_df.empty:
            y_min = float(draw_df["DrawdownPct"].min())
            draw_chart = (
                alt.Chart(draw_df)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title=""),
                    y=alt.Y(
                        "DrawdownPct:Q",
                        title="",
                        scale=alt.Scale(domain=[y_min, 0]),
                        axis=alt.Axis(format=".0f", labelExpr="datum.value + '%'"),
                    ),
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(draw_chart, use_container_width=True)
        else:
            st.caption("No drawdown data available.")

    st.markdown("#### Performance Metrics")
    summary = result.get("summary", {}) or {}
    pct_metrics = {
        "CAGR",
        "volatility",
        "max_drawdown",
        "worst_day",
        "worst_month",
        "VaR_95",
        "CVaR_95",
    }
    rows = []
    for metric, raw in summary.items():
        if isinstance(raw, (int, float)):
            if metric in pct_metrics:
                val = f"{float(raw) * 100:.2f}%"
            else:
                val = f"{float(raw):.4f}"
        else:
            val = raw
        rows.append({"Metric": metric, "Value": val})
    summary_df = pd.DataFrame(rows, columns=["Metric", "Value"])
    summary_styler = (
        summary_df.style.hide(axis="index").set_table_styles(
            [
                {"selector": "th.col0, td.col0", "props": [("text-align", "left")]},
                {"selector": "th.col1, td.col1", "props": [("text-align", "center")]},
            ]
        )
    )
    st.markdown(f'<div class="ii-table-wrap">{summary_styler.to_html()}</div>', unsafe_allow_html=True)

    scenario = result.get("scenario_results")
    if scenario:
        st.markdown("#### Monte Carlo Scenario Summary")
        end_pct = scenario.get("ending_value_percentiles", {})
        dd_pct = scenario.get("max_drawdown_percentiles", {})
        mc_rows = [
            {"Statistic": "Ending Value p05 (very conservative outcome; only 5% of runs end lower)", "Value": f"${float(end_pct.get('p05') or 0.0):,.0f}"},
            {"Statistic": "Ending Value p50 (middle outcome; typical expected ending value)", "Value": f"${float(end_pct.get('p50') or 0.0):,.0f}"},
            {"Statistic": "Ending Value p95 (strong outcome; only 5% of runs end higher)", "Value": f"${float(end_pct.get('p95') or 0.0):,.0f}"},
            {"Statistic": "Max Drawdown p05 (severe drop scenario in bad outcomes)", "Value": f"{float(dd_pct.get('p05') or 0.0) * 100:.2f}%"},
            {"Statistic": "Max Drawdown p50 (typical deepest drop during the period)", "Value": f"{float(dd_pct.get('p50') or 0.0) * 100:.2f}%"},
            {"Statistic": "Max Drawdown p95 (milder drop scenario in better outcomes)", "Value": f"{float(dd_pct.get('p95') or 0.0) * 100:.2f}%"},
            {"Statistic": "Probability of Loss (chance portfolio ends below starting capital)", "Value": f"{float(scenario.get('probability_of_loss') or 0.0) * 100:.2f}%"},
        ]
        mc_df = pd.DataFrame(mc_rows, columns=["Statistic", "Value"])
        mc_styler = (
            mc_df.style.hide(axis="index").set_table_styles(
                [
                    {"selector": "th.col0, td.col0", "props": [("text-align", "left")]},
                    {"selector": "th.col1, td.col1", "props": [("text-align", "center")]},
                ]
            )
        )
        st.markdown(f'<div class="ii-table-wrap">{mc_styler.to_html()}</div>', unsafe_allow_html=True)


def _show_decision_intelligence_tab() -> None:
    st.markdown("## Decision Intelligence")

    quality_df, _qerr = read_parquet_safe(QUALITY_CACHE_PATH)
    regime_df, _rerr = read_parquet_safe(REGIME_CACHE_PATH)
    risk_df, _risk_err = read_parquet_safe(RISK_CACHE_PATH)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Quality Rows", int(len(quality_df)) if quality_df is not None else 0)
    with c2:
        latest_regime = "Unknown"
        if regime_df is not None and not regime_df.empty and "RegimeLabel" in regime_df.columns:
            latest_regime = str(regime_df.iloc[-1]["RegimeLabel"])
        st.metric("Latest Regime", latest_regime)
    with c3:
        latest_risk = "Unknown"
        if risk_df is not None and not risk_df.empty and "RiskLevel" in risk_df.columns:
            latest_risk = str(risk_df.iloc[-1]["RiskLevel"])
        st.metric("Latest Risk", latest_risk)

    if quality_df is None and regime_df is None and risk_df is None:
        st.warning("Model artifacts not found. Run pipeline with run_models=True.")
        return

    if quality_df is not None and not quality_df.empty:
        st.markdown("#### Quality Score Distribution")
        qtmp = quality_df.copy()
        if "QualityScore" in qtmp.columns:
            qtmp["QualityScore"] = pd.to_numeric(qtmp["QualityScore"], errors="coerce")
            bins = pd.cut(
                qtmp["QualityScore"],
                bins=[0, 20, 40, 60, 80, 100],
                labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
                include_lowest=True,
            )
            dist = bins.value_counts().sort_index().reset_index()
            dist.columns = ["Bucket", "Count"]
            st.bar_chart(dist.set_index("Bucket"))
        st.markdown("#### Top Quality Entities")
        top_q = quality_df.sort_values("QualityScore", ascending=False).head(30)
        st.dataframe(top_q, use_container_width=True, hide_index=True)

    if regime_df is not None and not regime_df.empty and {"Date", "RegimeLabel", "ConfidenceScore"}.issubset(set(regime_df.columns)):
        st.markdown("#### Regime Timeline")
        rtmp = regime_df.copy()
        rtmp["Date"] = pd.to_datetime(rtmp["Date"], errors="coerce")
        rtmp = rtmp.dropna(subset=["Date"]).sort_values("Date")
        label_map = {"Risk Off": -1, "Neutral": 0, "Risk On": 1}
        rtmp["RegimeCode"] = rtmp["RegimeLabel"].map(label_map).fillna(0)
        chart = (
            alt.Chart(rtmp.tail(500))
            .mark_line()
            .encode(
                x=alt.X("Date:T", title=""),
                y=alt.Y("RegimeCode:Q", title="Regime (-1 Off, 0 Neutral, 1 On)"),
                color=alt.Color("RegimeLabel:N", legend=alt.Legend(title="")),
            )
            .properties(height=260)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    if risk_df is not None and not risk_df.empty and {"Date", "RiskScore", "RiskLevel"}.issubset(set(risk_df.columns)):
        st.markdown("#### Systemic Risk Timeline")
        ktmp = risk_df.copy()
        ktmp["Date"] = pd.to_datetime(ktmp["Date"], errors="coerce")
        ktmp["RiskScore"] = pd.to_numeric(ktmp["RiskScore"], errors="coerce")
        ktmp = ktmp.dropna(subset=["Date", "RiskScore"]).sort_values("Date")
        risk_chart = (
            alt.Chart(ktmp.tail(500))
            .mark_line(color="#ff9f43")
            .encode(
                x=alt.X("Date:T", title=""),
                y=alt.Y("RiskScore:Q", title="Risk Score"),
            )
            .properties(height=260)
            .interactive()
        )
        st.altair_chart(risk_chart, use_container_width=True)
        st.markdown("#### Recent Risk Signals")
        st.dataframe(ktmp.tail(30), use_container_width=True, hide_index=True)

    st.markdown("#### Model Registry")
    try:
        import json

        with open("data/model_registry.json", "r", encoding="utf-8") as f:
            registry = json.load(f)
        st.json(registry)
    except Exception:
        st.caption("`data/model_registry.json` not available.")

    st.markdown("#### Model Health")
    try:
        import json

        with open("data/model_health_report.json", "r", encoding="utf-8") as f:
            health = json.load(f)
        st.json(health)
    except Exception:
        st.caption("`data/model_health_report.json` not available.")


def _show_explainability_tab() -> None:
    st.markdown("## Explainability and Evidence")

    import json

    qexp, _qe = read_parquet_safe(QUALITY_EXPLAIN_PATH)
    qbase, _qb = read_parquet_safe(QUALITY_CACHE_PATH)
    revid, _re = read_parquet_safe(REGIME_EVIDENCE_PATH)
    revbase, _rb = read_parquet_safe(REGIME_CACHE_PATH)
    kevid, _ke = read_parquet_safe(RISK_EVIDENCE_PATH)
    kbase, _kb = read_parquet_safe(RISK_CACHE_PATH)
    fdf = _load_fundamentals_union()

    # Section A: Entity Explainability
    st.markdown("### A. Entity Explainability")
    if qexp is None or qexp.empty:
        st.warning("Quality explanations cache missing. Showing fallback from quality scores cache when available.")
        if qbase is None or qbase.empty:
            st.caption("No quality score data available.")
        else:
            st.dataframe(qbase.head(50), use_container_width=True, hide_index=True)
    else:
        qexp = qexp.copy()
        qexp["Ticker"] = qexp["Ticker"].astype(str).str.upper().str.strip()
        tickers = sorted(qexp["Ticker"].dropna().unique().tolist())
        t = st.selectbox("Ticker", options=tickers, index=0, key="xpl_ticker")
        row = qexp[qexp["Ticker"] == t].head(1).iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("QualityScore", f"{float(row.get('QualityScore', 0.0)):.2f}")
        with c2:
            st.metric("QualityTier", str(row.get("QualityTier", "Unknown")))
        with c3:
            st.metric("FeatureAsOfDate", str(row.get("FeatureAsOfDate", "Unknown")))
        st.caption(f"Top positive drivers: {row.get('TopPositiveDrivers', 'None')}")
        st.caption(f"Top negative drivers: {row.get('TopNegativeDrivers', 'None')}")

        contrib_map = {}
        try:
            contrib_map = json.loads(str(row.get("ContributionJSON", "{}")))
        except Exception:
            contrib_map = {}
        cdf = pd.DataFrame([{"Feature": k, "SignedContribution": float(v)} for k, v in contrib_map.items()])
        if not cdf.empty:
            cdf["AbsContribution"] = cdf["SignedContribution"].abs()
            cdf = cdf.sort_values("AbsContribution", ascending=False).drop(columns=["AbsContribution"])
            st.dataframe(cdf, use_container_width=True, hide_index=True)
            top_feats = cdf["Feature"].head(3).tolist()
            if not fdf.empty:
                raw_vals = []
                fm = fdf.set_index("Ticker")
                if t in fm.index:
                    for feat in top_feats:
                        base_feat = feat.replace("_stability", "")
                        val = fm.loc[t].get(base_feat) if base_feat in fm.columns else None
                        raw_vals.append({"Feature": feat, "RawValue": val})
                    st.markdown("#### Evidence Card")
                    st.dataframe(pd.DataFrame(raw_vals), use_container_width=True, hide_index=True)

    # Section B: Regime Evidence
    st.markdown("### B. Operating Environment Evidence")
    src_regime = revid if revid is not None and not revid.empty else revbase
    if src_regime is None or src_regime.empty:
        st.caption("No regime evidence available.")
    else:
        r = src_regime.copy()
        r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
        r = r.dropna(subset=["Date"]).sort_values("Date")
        date_opts = [d.strftime("%Y-%m-%d") for d in r["Date"].tolist()]
        sel = st.selectbox("Regime date", options=date_opts, index=len(date_opts) - 1, key="xpl_regime_date")
        rr = r[r["Date"] == pd.to_datetime(sel)].tail(1).iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RegimeLabel", str(rr.get("RegimeLabel", "Unknown")))
        with c2:
            if "ConfidenceScore" in rr:
                st.metric("ConfidenceScore", f"{float(rr.get('ConfidenceScore', 0.0)):.2f}")

        st.caption(f"RuleTriggered: {rr.get('RuleTriggered', 'N/A')}")
        st.caption(f"{rr.get('ShortExplanation', 'No explanation available.')}")

        evjson = rr.get("EvidencePointsJSON", "{}")
        try:
            evmap = json.loads(str(evjson))
        except Exception:
            evmap = {}
        if evmap:
            edt = pd.DataFrame([{"Indicator": k, "Value": v} for k, v in evmap.items()])
            st.dataframe(edt, use_container_width=True, hide_index=True)

    # Section C: Systemic Risk Evidence
    st.markdown("### C. Systemic Risk Evidence")
    src_risk = kevid if kevid is not None and not kevid.empty else kbase
    if src_risk is None or src_risk.empty:
        st.caption("No risk evidence available.")
    else:
        k = src_risk.copy()
        k["Date"] = pd.to_datetime(k["Date"], errors="coerce")
        k = k.dropna(subset=["Date"]).sort_values("Date")
        date_opts = [d.strftime("%Y-%m-%d") for d in k["Date"].tolist()]
        sel = st.selectbox("Risk date", options=date_opts, index=len(date_opts) - 1, key="xpl_risk_date")
        lookback_days = st.slider("Last N days", min_value=30, max_value=365, value=120, step=10, key="xpl_risk_lookback")
        kr = k[k["Date"] == pd.to_datetime(sel)].tail(1).iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            if "RiskScore" in kr:
                st.metric("RiskScore", f"{float(kr.get('RiskScore', 0.0)):.2f}")
        with c2:
            st.metric("RiskLevel", str(kr.get("RiskLevel", "Unknown")))
        if "TopRiskDrivers" in kr:
            st.caption(f"TopRiskDrivers: {kr.get('TopRiskDrivers')}")
        if "ShortExplanation" in kr:
            st.caption(str(kr.get("ShortExplanation")))

        evjson = kr.get("EvidencePointsJSON", "{}")
        try:
            evmap = json.loads(str(evjson))
        except Exception:
            evmap = {}
        if evmap:
            edt = pd.DataFrame([{"Indicator": k2, "Value": v2} for k2, v2 in evmap.items()])
            st.dataframe(edt, use_container_width=True, hide_index=True)

        ktail = k.tail(int(lookback_days)).copy()
        if "RiskScore" in ktail.columns:
            st.line_chart(ktail.set_index("Date")[["RiskScore"]])

        ind_options = [c for c in ["VolatilityExpansion", "RapidDrawdown", "CorrelationSpike", "YieldCurveInversion", "RateShock"] if c in evmap]
        if ind_options:
            indicator = st.selectbox("Underlying indicator", options=ind_options, index=0, key="xpl_indicator")
            ser_rows = []
            for _, row in ktail.iterrows():
                try:
                    m = json.loads(str(row.get("EvidencePointsJSON", "{}")))
                    ser_rows.append({"Date": row["Date"], "Value": m.get(indicator)})
                except Exception:
                    continue
            sdt = pd.DataFrame(ser_rows).dropna()
            if not sdt.empty:
                st.line_chart(sdt.set_index("Date")[["Value"]].rename(columns={"Value": indicator}))


_show_signal_banner()

stock_tab, fi_tab, sim_tab, di_tab, ex_tab = st.tabs(
    ["Stock Screener", "Bond & Treasury Screener", "Portfolio Decision Simulator", "Decision Intelligence", "Explainability and Evidence"]
)
with stock_tab:
    _show_stock_tab()
with fi_tab:
    _show_fixed_income_tab()
with sim_tab:
    _show_portfolio_simulator_tab()
with di_tab:
    _show_decision_intelligence_tab()
with ex_tab:
    _show_explainability_tab()
