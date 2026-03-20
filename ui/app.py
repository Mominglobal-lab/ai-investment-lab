from __future__ import annotations

from dataclasses import dataclass
import html
import sys
import textwrap
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    from st_aggrid.shared import JsCode
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    JsCode = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.cache_manager import get_cache_status, read_parquet_safe
from data_pipeline.data_fetcher import FI_SCHEMA_COLUMNS, SCHEMA_COLUMNS
from reports.decision_brief import generate_decision_brief
from simulation.portfolio_simulator import simulate_portfolio

st.set_page_config(page_title="Investment Lab", layout="wide")

STOCK_UNIVERSE_OPTIONS = ["S&P 500", "Nasdaq 100"]
FI_UNIVERSE_OPTIONS = ["US Treasuries", "Bond ETFs"]
SIM_BENCHMARK_OPTIONS = ["SPY", "QQQ", "IWM", "DIA"]
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
QUALITY_UNCERTAINTY_PATH = "data/quality_uncertainty_cache.parquet"
REGIME_PROB_PATH = "data/regime_probabilities_cache.parquet"
RISK_UNCERTAINTY_PATH = "data/risk_uncertainty_cache.parquet"
DRIFT_SIGNALS_PATH = "data/drift_signals_cache.parquet"
DRIFT_SIGNALS_HISTORY_PATH = "data/drift_signals_history.parquet"
DRIFT_REPORT_PATH = "data/drift_report.json"
ALERT_LOG_PATH = "data/alert_log.parquet"
MONITORING_HEALTH_PATH = "data/monitoring_health_report.json"


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
            border-top: 1px solid #ccd8e8 !important;
            border-bottom: 1px solid #ccd8e8 !important;
        }

        [data-testid="stDataFrame"] [role="columnheader"] {
            border-bottom: 1px solid #b7c8dc !important;
            border-right: 1px solid #c2d1e4 !important;
        }

        [data-testid="stDataFrame"] [role="gridcell"] {
            border-bottom: 1px solid #c2d1e4 !important;
            border-right: 1px solid #c2d1e4 !important;
            box-shadow: inset 0 -1px 0 #c2d1e4;
        }

        [data-testid="stDataFrame"] [role="columnheader"]:last-child,
        [data-testid="stDataFrame"] [role="gridcell"]:last-child {
            border-right: none !important;
        }

        [data-testid="stDataFrame"] td:has(span[title="None"]),
        [data-testid="stDataFrame"] td:has(div[title="None"]) {
            color: #7f8fa8 !important;
        }

        [data-testid="stJson"] {
            border: 1px solid #b8c6dc;
            border-radius: 12px;
            background: #ffffff !important;
            padding: 0.35rem 0.5rem;
        }

        [data-testid="stJson"] *,
        [data-testid="stJson"] span,
        [data-testid="stJson"] div {
            color: #0a0f18 !important;
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

        .ii-table-wrap table,
        .ii-table-wrap table.dataframe {
            width: 100%;
            border-collapse: collapse !important;
        }

        .ii-table-wrap thead th {
            color: #05070b !important;
            font-weight: 900 !important;
            background: #edf3fb !important;
            border-bottom: 1px solid #b7c8dc !important;
            text-align: left !important;
        }

        .ii-table-wrap tbody tr td,
        .ii-table-wrap table.dataframe tbody tr td {
            border-bottom: 1px solid #c2d1e4 !important;
        }

        .ii-table-wrap tbody tr:last-child td,
        .ii-table-wrap table.dataframe tbody tr:last-child td {
            border-bottom: 1px solid #c2d1e4 !important;
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


def _load_explainability_feature_union() -> pd.DataFrame:
    paths = [
        "data/fundamentals_cache.parquet",
        "data/fundamentals_cache_sp500.parquet",
        "data/fundamentals_cache_nasdaq100.parquet",
    ]
    fund_frames: list[pd.DataFrame] = []
    wanted_fund_cols = [
        "Ticker",
        "ROE",
        "EBITDA_Margin",
        "Revenue_Growth_YoY_Pct",
        "Earnings_Growth_Pct",
        "FreeCashFlow_Margin",
        "MarketCap",
    ]
    for p in paths:
        df, _err = read_parquet_safe(p)
        if df is None or df.empty or "Ticker" not in df.columns:
            continue
        keep = [c for c in wanted_fund_cols if c in df.columns]
        if not keep:
            continue
        f = df[keep].copy()
        f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
        for c in [x for x in keep if x != "Ticker"]:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        fund_frames.append(f)

    fund_df = pd.concat(fund_frames, axis=0, ignore_index=True) if fund_frames else pd.DataFrame(columns=["Ticker"])
    if not fund_df.empty:
        fund_df = fund_df.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last")

    px_df, _perr = read_parquet_safe(PRICES_CACHE_PATH)
    if px_df is None or px_df.empty or not {"Ticker", "Date", "AdjClose"}.issubset(set(px_df.columns)):
        return fund_df

    px = px_df.copy()
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()
    px["Date"] = pd.to_datetime(px["Date"], errors="coerce")
    px["AdjClose"] = pd.to_numeric(px["AdjClose"], errors="coerce")
    px = px.dropna(subset=["Ticker", "Date", "AdjClose"]).sort_values(["Ticker", "Date"])

    rows: list[dict[str, object]] = []
    for ticker, g in px.groupby("Ticker"):
        s = g.set_index("Date")["AdjClose"].sort_index()
        r1 = s.pct_change()
        vol_63d = float(r1.rolling(63).std().iloc[-1] * (252.0 ** 0.5)) if len(r1) > 63 else float("nan")
        dd_252d = float("nan")
        if len(s) >= 30:
            roll_max = s.rolling(252, min_periods=30).max()
            dd = (s / roll_max) - 1.0
            if not dd.empty:
                dd_252d = float(dd.iloc[-1])
        rows.append({"Ticker": ticker, "Volatility_63D": vol_63d, "Drawdown_252D": dd_252d})

    price_feat_df = pd.DataFrame(rows)
    if fund_df.empty:
        return price_feat_df
    return fund_df.merge(price_feat_df, on="Ticker", how="outer")


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


def _render_styled_table(df: pd.DataFrame) -> None:
    show = df.copy()
    numeric_cols = show.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        show[numeric_cols] = show[numeric_cols].round(2)
    try:
        st.dataframe(show, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(show, use_container_width=True)


def _render_sortable_centered_table(df: pd.DataFrame, center_cols: list[str], page_size: int | None = None) -> None:
    show = df.copy()
    numeric_cols = show.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        show[numeric_cols] = show[numeric_cols].round(2)

    valid_center_cols = [c for c in center_cols if c in show.columns]
    if AgGrid is not None and GridOptionsBuilder is not None:
        gb = GridOptionsBuilder.from_dataframe(show)
        gb.configure_default_column(sortable=True, resizable=True, flex=1, minWidth=120)
        for col in valid_center_cols:
            gb.configure_column(
                col,
                cellStyle={"textAlign": "center"},
                headerClass="di-header-center",
                cellClass="di-cell-center",
            )
        grid_options = gb.build()
        grid_options["suppressHorizontalScroll"] = False
        if page_size is not None and int(page_size) > 0:
            grid_options["pagination"] = True
            grid_options["paginationPageSize"] = int(page_size)
        row_h = 36
        header_h = 40
        grid_options["rowHeight"] = row_h
        grid_options["headerHeight"] = header_h
        visible_rows = (
            min(max(1, len(show)), int(page_size))
            if page_size is not None and int(page_size) > 0
            else max(1, len(show))
        )
        grid_height = int(header_h + (max(1, visible_rows) * row_h) + 2)
        AgGrid(
            show,
            gridOptions=grid_options,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            theme="streamlit",
            height=grid_height,
            width="100%",
            custom_css={
                ".ag-header-cell.di-header-center .ag-header-cell-label": {
                    "justify-content": "center"
                },
                ".ag-cell.di-cell-center .ag-cell-wrapper": {
                    "justify-content": "center"
                }
            },
        )
        return

    if valid_center_cols:
        styled = show.style.set_properties(subset=valid_center_cols, **{"text-align": "center"})
        try:
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(styled, use_container_width=True)
    else:
        try:
            st.dataframe(show, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(show, use_container_width=True)


def _render_sortable_all_but_first_table(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    center_cols = cols[1:] if len(cols) > 1 else []
    _render_sortable_centered_table(df, center_cols)


def _render_sortable_first_col_centered_table(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    center_cols = cols[:1] if cols else []
    _render_sortable_centered_table(df, center_cols)


def _render_stock_ticker_detail_card(d: pd.Series, ticker: str) -> None:
    def _as_num(value: object) -> float | None:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(v) else float(v)

    def _fmt_plain(value: object, decimals: int = 2) -> str:
        v = _as_num(value)
        return "N/A" if v is None else f"{v:.{decimals}f}"

    def _fmt_pct(value: object, decimals: int = 2, scale: float = 1.0) -> str:
        v = _as_num(value)
        return "N/A" if v is None else f"{(v * scale):.{decimals}f}%"

    def _fmt_money(value: object, decimals: int = 2) -> str:
        v = _as_num(value)
        return "N/A" if v is None else f"${v:,.{decimals}f}"

    company = str(d.get("Company") or ticker)
    sector = str(d.get("Sector") or "N/A")
    quality_tier = str(d.get("QualityTier") or "Unknown")
    quality_score = _as_num(d.get("QualityScore"))
    quality_score_txt = "N/A" if quality_score is None else f"{quality_score:.1f} / 100"
    quality_bar = 0.0 if quality_score is None else max(0.0, min(100.0, quality_score))

    tier_colors = {"Strong": "#2f9e44", "Neutral": "#d4a017", "Weak": "#d94841"}
    tier_bg = tier_colors.get(quality_tier, "#6b7280")

    market_cap_b = _as_num(d.get("MarketCap"))
    market_cap_txt = "N/A" if market_cap_b is None else f"${market_cap_b / 1e9:.1f}B"

    st.markdown(
        textwrap.dedent(
            f"""
        <style>
        .stock-detail-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            color: var(--text);
            width: 100%;
            max-width: none;
            box-sizing: border-box;
            box-shadow: var(--shadow);
        }}
        .stock-detail-head {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
        }}
        .stock-detail-title {{ font-size: 1.2rem; font-weight: 700; line-height: 1.2; }}
        .stock-detail-sub {{ color: var(--muted); font-size: 1.1rem; font-weight: 600; }}
        .stock-detail-badge {{
            background: {tier_bg};
            color: #f7f7f7;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            white-space: nowrap;
        }}
        .stock-detail-badge-wrap {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        .stock-detail-badge-info {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text);
            background: var(--surface-2);
            border: 1px solid var(--border);
            cursor: help;
            user-select: none;
        }}
        .stock-detail-price {{
            margin-top: 10px;
            font-size: 2.6rem;
            font-weight: 800;
        }}
        .stock-detail-price-sub {{
            color: var(--muted);
            font-size: 0.95rem;
            margin-left: 8px;
            font-weight: 600;
        }}
        .stock-detail-divider {{
            border-top: 1px solid var(--border);
            margin: 14px 0 10px 0;
        }}
        .stock-detail-section-title {{
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .stock-detail-info {{
            display: inline-block;
            margin-left: 8px;
            width: 18px;
            height: 18px;
            line-height: 18px;
            border-radius: 999px;
            text-align: center;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text);
            background: var(--surface-2);
            border: 1px solid var(--border);
            cursor: help;
            vertical-align: middle;
        }}
        .stock-detail-tip {{
            position: relative;
        }}
        .stock-detail-tip:hover::after {{
            content: attr(data-tooltip);
            position: absolute;
            left: 0;
            bottom: calc(100% + 10px);
            background: #0a1324;
            color: #e8f1ff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            width: min(320px, calc(100vw - 32px));
            white-space: normal;
            text-align: left;
            line-height: 1.35;
            font-size: 0.82rem;
            font-weight: 500;
            overflow-wrap: break-word;
            z-index: 9999;
            box-shadow: var(--shadow);
            pointer-events: none;
        }}
        .stock-detail-tip-right:hover::after {{
            left: auto;
            right: 0;
        }}
        .stock-detail-tip-below:hover::after {{
            bottom: auto;
            top: calc(100% + 10px);
        }}
        .stock-detail-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
        .stock-detail-grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .stock-detail-chip {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 14px;
        }}
        .stock-detail-chip .k {{ color: var(--muted); font-size: 1.05rem; font-weight: 600; }}
        .stock-detail-chip .v {{ color: var(--text); font-size: 1.2rem; font-weight: 800; margin-top: 2px; }}
        .stock-quality-row {{ display: flex; align-items: center; gap: 10px; }}
        .stock-quality-bar-wrap {{
            flex: 1;
            height: 10px;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            overflow: hidden;
        }}
        .stock-quality-bar {{
            height: 100%;
            width: {quality_bar:.1f}%;
            background: linear-gradient(90deg, #20c997, #2b8a3e);
        }}
        .stock-quality-score {{ font-size: 1.0rem; font-weight: 700; }}
        .stock-quality-pill {{
            background: var(--surface-2);
            border-radius: 999px;
            padding: 3px 12px;
            font-weight: 700;
            color: var(--text);
        }}
        </style>
        <div class="stock-detail-card">
            <div class="stock-detail-head">
                <div>
                    <div class="stock-detail-title">{html.escape(company)}</div>
                    <div class="stock-detail-sub">{html.escape(str(ticker).upper())} - {html.escape(sector)}</div>
                </div>
                <div class="stock-detail-badge-wrap">
                    <span class="stock-detail-badge-info stock-detail-tip stock-detail-tip-right stock-detail-tip-below" data-tooltip="Quality Tier is derived from the Quality Score (0-100), a weighted composite of percentile-ranked Revenue Growth, EBITDA Margin, ROE, Free Cash Flow Margin, Volatility Stability, and Drawdown Stability. Tier cutoffs: Strong (>=67), Neutral (34-66), Weak (<34).">i</span>
                    <div class="stock-detail-badge">{html.escape(quality_tier)}</div>
                </div>
            </div>
            <div class="stock-detail-price">
                {_fmt_money(d.get("Close"), 2)}
                <span class="stock-detail-price-sub">close - Mkt Cap {market_cap_txt}</span>
            </div>
            <div class="stock-detail-divider"></div>
            <div class="stock-detail-section-title">Growth</div>
            <div class="stock-detail-grid two">
                <div class="stock-detail-chip"><div class="k">Revenue growth</div><div class="v">{_fmt_pct(d.get("Revenue_Growth_YoY_Pct"), 2, 1.0)}</div></div>
                <div class="stock-detail-chip"><div class="k">Earnings growth</div><div class="v">{_fmt_pct(d.get("Earnings_Growth_Pct"), 2, 1.0)}</div></div>
            </div>
            <div class="stock-detail-divider"></div>
            <div class="stock-detail-section-title">Valuation</div>
            <div class="stock-detail-grid">
                <div class="stock-detail-chip"><div class="k">P/E</div><div class="v">{_fmt_plain(d.get("PE_Ratio"), 2)}</div></div>
                <div class="stock-detail-chip"><div class="k">PEG</div><div class="v">{_fmt_plain(d.get("PEG_Ratio"), 2)}</div></div>
                <div class="stock-detail-chip"><div class="k">Rule of 40</div><div class="v">{_fmt_plain(d.get("Rule_of_40"), 1)}</div></div>
            </div>
            <div class="stock-detail-divider"></div>
            <div class="stock-detail-section-title">Profitability</div>
            <div class="stock-detail-grid two">
                <div class="stock-detail-chip"><div class="k">EBITDA margin</div><div class="v">{_fmt_pct(d.get("EBITDA_Margin"), 2, 100.0)}</div></div>
                <div class="stock-detail-chip"><div class="k">ROE</div><div class="v">{_fmt_pct(d.get("ROE"), 2, 100.0)}</div></div>
            </div>
            <div class="stock-detail-divider"></div>
            <div class="stock-detail-section-title">Quality
                <span class="stock-detail-info stock-detail-tip" data-tooltip="Quality Score is a 0-100 composite based on percentile-ranked Revenue Growth, EBITDA Margin, ROE, Free Cash Flow Margin, Volatility Stability, and Drawdown Stability. Higher scores indicate stronger relative quality within the screened universe.">i</span>
            </div>
            <div class="stock-quality-row">
                <div class="stock-quality-bar-wrap"><div class="stock-quality-bar"></div></div>
                <div class="stock-quality-score">{quality_score_txt}</div>
                <div class="stock-quality-pill">{html.escape(quality_tier)}</div>
            </div>
        </div>
        """,
        ),
        unsafe_allow_html=True,
    )


def _render_fixed_income_detail_card(row: pd.Series, symbol: str) -> None:
    def _as_num(value: object) -> float | None:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(v) else float(v)

    def _fmt_plain(value: object, decimals: int = 2) -> str:
        v = _as_num(value)
        return "N/A" if v is None else f"{v:.{decimals}f}"

    def _fmt_pct(value: object, decimals: int = 2) -> str:
        v = _as_num(value)
        return "N/A" if v is None else f"{v:.{decimals}f}%"

    def _fmt_money(value: object, decimals: int = 2) -> str:
        v = _as_num(value)
        return "N/A" if v is None else f"${v:,.{decimals}f}"

    name = str(row.get("Name") or symbol)
    universe = str(row.get("Universe") or "N/A")
    inst_type = str(row.get("Type") or "Instrument")
    maturity_bucket = str(row.get("Maturity_Bucket") or "N/A")
    aum = _as_num(row.get("AUM"))
    aum_txt = "N/A" if aum is None else f"${aum / 1e9:.2f}B"
    duration = _as_num(row.get("Duration_Years"))
    price_impact = "N/A" if duration is None else f"{-duration:.2f}%"

    yield_pct = _as_num(row.get("Yield_Pct"))
    if yield_pct is None:
        rating_label = "Unknown"
        rating_bg = "#6b7280"
    elif yield_pct >= 6.0:
        rating_label = "High Yield"
        rating_bg = "#d94841"
    elif yield_pct >= 3.0:
        rating_label = "Income"
        rating_bg = "#2f9e44"
    else:
        rating_label = "Low Yield"
        rating_bg = "#3b82f6"

    st.markdown(
        textwrap.dedent(
            f"""
        <style>
        .fi-detail-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            color: var(--text);
            width: 100%;
            max-width: none;
            box-sizing: border-box;
            box-shadow: var(--shadow);
        }}
        .fi-detail-head {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
        }}
        .fi-detail-title {{ font-size: 1.2rem; font-weight: 700; line-height: 1.2; }}
        .fi-detail-sub {{ color: var(--muted); font-size: 1.1rem; font-weight: 600; }}
        .fi-detail-badge {{
            background: {rating_bg};
            color: #f7f7f7;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            white-space: nowrap;
        }}
        .fi-detail-price {{
            margin-top: 10px;
            font-size: 2.3rem;
            font-weight: 800;
        }}
        .fi-detail-price-sub {{
            color: var(--muted);
            font-size: 0.95rem;
            margin-left: 8px;
            font-weight: 600;
        }}
        .fi-detail-divider {{
            border-top: 1px solid var(--border);
            margin: 14px 0 10px 0;
        }}
        .fi-detail-section-title {{
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .fi-detail-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
        .fi-detail-grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .fi-detail-chip {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 14px;
        }}
        .fi-detail-chip .k {{ color: var(--muted); font-size: 1.05rem; font-weight: 600; }}
        .fi-detail-chip .v {{ color: var(--text); font-size: 1.2rem; font-weight: 800; margin-top: 2px; }}
        </style>
        <div class="fi-detail-card">
            <div class="fi-detail-head">
                <div>
                    <div class="fi-detail-title">{html.escape(name)}</div>
                    <div class="fi-detail-sub">{html.escape(str(symbol).upper())} - {html.escape(universe)}</div>
                </div>
                <div class="fi-detail-badge">{html.escape(rating_label)}</div>
            </div>
            <div class="fi-detail-price">
                {_fmt_money(row.get("Price"), 2)}
                <span class="fi-detail-price-sub">{html.escape(inst_type)} - {html.escape(maturity_bucket)}</span>
            </div>
            <div class="fi-detail-divider"></div>
            <div class="fi-detail-section-title">Yield & Duration</div>
            <div class="fi-detail-grid two">
                <div class="fi-detail-chip"><div class="k">Yield</div><div class="v">{_fmt_pct(row.get("Yield_Pct"), 2)}</div></div>
                <div class="fi-detail-chip"><div class="k">Duration (yrs)</div><div class="v">{_fmt_plain(row.get("Duration_Years"), 2)}</div></div>
            </div>
            <div class="fi-detail-divider"></div>
            <div class="fi-detail-section-title">Costs & Size</div>
            <div class="fi-detail-grid two">
                <div class="fi-detail-chip"><div class="k">Expense ratio</div><div class="v">{_fmt_pct(row.get("Expense_Ratio_Pct"), 2)}</div></div>
                <div class="fi-detail-chip"><div class="k">AUM</div><div class="v">{aum_txt}</div></div>
            </div>
            <div class="fi-detail-divider"></div>
            <div class="fi-detail-section-title">Rate Shock</div>
            <div class="fi-detail-grid">
                <div class="fi-detail-chip"><div class="k">Estimated price impact for +100 bps</div><div class="v">{price_impact}</div></div>
                <div class="fi-detail-chip"><div class="k">Maturity bucket</div><div class="v">{html.escape(maturity_bucket)}</div></div>
                <div class="fi-detail-chip"><div class="k">Instrument type</div><div class="v">{html.escape(inst_type)}</div></div>
            </div>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def _render_explainability_table(df: pd.DataFrame, *, center_last_n: int = 1) -> None:
    show = df.copy()
    cols = list(show.columns)
    n = max(1, int(center_last_n))
    centered_cols = set(cols[-min(n, len(cols)) :]) if cols else set()
    numeric_cols: set[str] = set()
    percent_cols: set[str] = set()
    flag_cols: set[str] = set()
    percent_name_overrides = {
        "benchmarktrend_63d",
        "benchmarkvol_63d",
        "volatility_63d",
        "drawdown_252d",
    }
    for col in cols:
        lname = str(col).lower()
        raw = show[col]
        if lname.endswith("_flag") or "flag" in lname:
            flag_cols.add(col)
            continue
        if str(raw.dtype).lower() == "bool":
            flag_cols.add(col)
            continue
        raw = show[col]
        ser_num = pd.to_numeric(raw, errors="coerce")
        non_null = int(raw.notna().sum())
        if non_null == 0:
            continue
        # Columns that are effectively binary flags should be shown as True/False.
        uniq_vals = set(pd.Series(raw).dropna().astype(str).str.strip().str.lower().unique().tolist())
        if uniq_vals and uniq_vals.issubset({"0", "1", "true", "false"}):
            flag_cols.add(col)
            continue
        numeric_ratio = float(ser_num.notna().sum()) / float(non_null)
        if numeric_ratio >= 0.95:
            numeric_cols.add(col)
            if any(token in lname for token in ["pct", "percent", "prob", "confidence", "contribution"]) or lname in percent_name_overrides:
                percent_cols.add(col)

    if AgGrid is not None and GridOptionsBuilder is not None and JsCode is not None:
        gb = GridOptionsBuilder.from_dataframe(show)
        gb.configure_default_column(sortable=True, resizable=True, flex=1, minWidth=140)

        for col in cols:
            cfg: dict = {}
            if col in centered_cols:
                cfg["cellStyle"] = {"textAlign": "center"}
                cfg["headerClass"] = "xpl-header-center"
            if col in flag_cols:
                cfg["valueFormatter"] = JsCode(
                    "function(params){ if(params.value==null){return '';} if(params.value===true||params.value===1||params.value==='1'||String(params.value).toLowerCase()==='true'){return 'True';} return 'False'; }"
                )
            elif col in numeric_cols:
                if col in percent_cols:
                    cfg["valueFormatter"] = JsCode(
                        "function(params){ if(params.value==null){return '';} const v=Number(params.value); if(!Number.isFinite(v)){return '';} return (v*100).toFixed(2) + '%'; }"
                    )
                else:
                    cfg["valueFormatter"] = JsCode(
                        "function(params){ if(params.value==null){return '';} const v=Number(params.value); if(!Number.isFinite(v)){return '';} const a=Math.abs(v); if(a>0 && a<0.1){return v.toFixed(4);} return v.toFixed(2); }"
                    )
            gb.configure_column(col, **cfg)

        grid_options = gb.build()
        grid_options["suppressHorizontalScroll"] = False
        row_h = 36
        header_h = 40
        grid_options["rowHeight"] = row_h
        grid_options["headerHeight"] = header_h
        grid_height = int(header_h + (max(1, len(show)) * row_h) + 2)
        AgGrid(
            show,
            gridOptions=grid_options,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            theme="streamlit",
            height=grid_height,
            width="100%",
            custom_css={
                ".ag-header-cell.xpl-header-center .ag-header-cell-label": {
                    "justify-content": "center"
                }
            },
        )
        return

    display_df = show.copy()
    for col in flag_cols:
        display_df[col] = (
            pd.Series(show[col])
            .map(lambda v: "" if pd.isna(v) else ("True" if str(v).strip().lower() in {"1", "true"} else "False"))
        )
    for col in numeric_cols:
        if col in flag_cols:
            continue
        ser = pd.to_numeric(show[col], errors="coerce")
        if col in percent_cols:
            display_df[col] = ser.map(lambda v: "" if pd.isna(v) else f"{float(v) * 100.0:.2f}%")
        else:
            display_df[col] = ser.map(
                lambda v: ""
                if pd.isna(v)
                else (f"{float(v):.4f}" if (abs(float(v)) > 0 and abs(float(v)) < 0.1) else f"{float(v):.2f}")
            )
    try:
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(display_df, use_container_width=True)


def _format_evidence_points_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not {"Indicator", "Value"}.issubset(set(df.columns)):
        return df
    out = df.copy()
    pct_indicators = {"benchmarktrend_63d", "benchmarkvol_63d", "volatility_63d", "drawdown_252d"}
    values: list[object] = []
    for _, r in out.iterrows():
        indicator = str(r.get("Indicator", "")).strip()
        v = r.get("Value")
        il = indicator.lower()
        if il.endswith("_flag") or "flag" in il:
            if pd.isna(v):
                values.append("")
            else:
                s = str(v).strip().lower()
                values.append("True" if s in {"1", "true"} else "False")
            continue
        vn = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(vn):
            values.append(v)
            continue
        if il in pct_indicators:
            values.append(f"{float(vn) * 100.0:.2f}%")
        elif abs(float(vn)) > 0 and abs(float(vn)) < 0.1:
            values.append(f"{float(vn):.4f}")
        else:
            values.append(f"{float(vn):.2f}")
    out["Value"] = values
    return out


def _format_evidence_card_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not {"Feature", "RawValue"}.issubset(set(df.columns)):
        return df
    out = df.copy()
    percent_features = {
        "ebitda_margin",
        "roe",
        "revenue_growth_yoy_pct",
        "earnings_growth_pct",
        "freecashflow_margin",
        "volatility_63d",
        "drawdown_252d",
    }
    vals: list[object] = []
    for _, r in out.iterrows():
        feat = str(r.get("Feature", "")).replace("_stability", "").strip().lower()
        v = r.get("RawValue")
        vn = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(vn):
            vals.append("" if pd.isna(v) else v)
            continue
        if feat in percent_features:
            vals.append(f"{float(vn) * 100.0:.2f}%")
        elif abs(float(vn)) > 0 and abs(float(vn)) < 0.1:
            vals.append(f"{float(vn):.4f}")
        else:
            vals.append(f"{float(vn):.2f}")
    out["RawValue"] = vals
    return out


def _show_stock_tab() -> None:
    st.markdown("## Stock Screener")
    status_text = ""
    telemetry_text = ""
    st.caption("Universe")
    universe = st.selectbox(
        "Universe",
        STOCK_UNIVERSE_OPTIONS,
        index=0,
        key="stock_universe",
        label_visibility="collapsed",
    )

    cache_path, _health_report_path = _stock_paths(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=SCHEMA_COLUMNS)
    status_text = (
        ("No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days")
        + f" | {'Fresh' if status.is_fresh else 'Stale'} | {'Schema OK' if status.schema_ok else 'Schema mismatch'}"
    )

    stale_df, _err = read_parquet_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
    if stale_df is not None:
        stale_df = _ensure_stock_schema(stale_df)

    df_raw = stale_df
    if stale_df is None:
        if not status.exists:
            st.warning("Stock cache is missing. This UI now reads scheduled cache output only.")
        elif not status.schema_ok:
            st.warning("Stock cache schema is invalid. Rebuild the scheduled pipeline output before using this tab.")
        else:
            st.warning("Stock cache could not be loaded.")
        return
    telemetry_text = "Using cached stock data"
    if not status.is_fresh:
        telemetry_text = "Using stale stock cache until the next scheduled refresh"

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
    _render_sortable_centered_table(
        df_show,
        [
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
        ],
        page_size=20,
    )

    ticker = st.selectbox("Ticker details", options=[""] + df["Ticker"].astype(str).tolist(), index=0, key="stock_detail_ticker")
    if ticker:
        row = df[df["Ticker"] == ticker].head(1)
        if not row.empty:
            d = row.iloc[0]
            _render_stock_ticker_detail_card(d, ticker)
        else:
            st.caption("Ticker detail unavailable in cache.")

    st.divider()


def _show_fixed_income_tab() -> None:
    st.markdown("## Bond & Treasury Screener")
    status_text = ""
    telemetry_text = ""
    st.caption("Universe")
    universe = st.selectbox(
        "Universe",
        FI_UNIVERSE_OPTIONS,
        index=0,
        key="fi_universe",
        label_visibility="collapsed",
    )

    cache_path, health_report_path = _fi_paths(universe)
    status = get_cache_status(cache_path, MAX_AGE_DAYS, required_columns=FI_SCHEMA_COLUMNS)
    status_text = (
        ("No cache" if not status.exists else f"Cache age: {status.age_days:.2f} days")
        + f" | {'Fresh' if status.is_fresh else 'Stale'} | {'Schema OK' if status.schema_ok else 'Schema mismatch'}"
    )

    stale_df, _err = read_parquet_safe(cache_path) if (status.exists and status.schema_ok) else (None, None)
    df_raw = stale_df
    if stale_df is None:
        if not status.exists:
            st.warning("Fixed-income cache is missing. This UI now reads scheduled cache output only.")
        elif not status.schema_ok:
            st.warning("Fixed-income cache schema is invalid. Rebuild the scheduled pipeline output before using this tab.")
        else:
            st.warning("Fixed-income cache could not be loaded.")
        return
    telemetry_text = "Using cached fixed-income data"
    if not status.is_fresh:
        telemetry_text = "Using stale fixed-income cache until the next scheduled refresh"

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
    _render_sortable_centered_table(
        show,
        [
            "Name",
            "Universe",
            "Type",
            "Price",
            "Yield_Pct",
            "Duration_Years",
            "Maturity_Bucket",
            "Expense_Ratio_Pct",
            "AUM (B)",
        ],
        page_size=20,
    )

    detail_symbol = st.selectbox("Instrument details", options=[""] + df["Symbol"].astype(str).tolist(), index=0, key="fi_detail")
    if detail_symbol:
        row = df[df["Symbol"] == detail_symbol].head(1).iloc[0]
        _render_fixed_income_detail_card(row, detail_symbol)

    st.divider()


def _show_portfolio_simulator_tab() -> None:
    title_col, act1, act2 = st.columns([3.8, 1.0, 1.0], vertical_alignment="center")
    with title_col:
        st.markdown("## Portfolio Decision Simulator")
    with act1:
        run_clicked = st.button("Run Simulation", key="sim_run_btn", use_container_width=True)
    with act2:
        export_clicked = st.button("Export Decision Brief", key="sim_export_brief", use_container_width=True)

    price_status = get_cache_status(PRICES_CACHE_PATH, MAX_AGE_DAYS, required_columns=PRICE_SCHEMA_COLUMNS)
    if (not price_status.exists) or (not price_status.schema_ok):
        st.warning("Price cache not found or invalid. Run the scheduled pipeline before using the simulator.")
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
        benchmark = st.selectbox(
            "Benchmark",
            SIM_BENCHMARK_OPTIONS,
            index=0,
            key="sim_benchmark",
        )

    manual_weights_text = ""
    if weighting_mode == "Manual weights":
        manual_weights_text = st.text_input(
            "Manual weights (comma-separated, same order as tickers)",
            value="0.25, 0.25, 0.25, 0.25",
            key="sim_manual_weights",
        )

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
    _render_sortable_centered_table(summary_df, ["Value"])

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
        _render_sortable_centered_table(mc_df, ["Value"])


def _show_decision_intelligence_tab() -> None:
    st.markdown("## Decision Intelligence")
    st.caption("Artifacts are read from cache generated by the scheduled pipeline.")

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
        _render_sortable_centered_table(top_q, ["QualityScore", "QualityTier"])

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
            .properties(height=300, padding={"left": 8, "right": 12, "top": 8, "bottom": 34})
            .configure_axis(
                labelColor="#dce9ff",
                titleColor="#dce9ff",
                labelPadding=8,
                titlePadding=10,
                gridColor="#30445f",
                tickColor="#496184",
                domainColor="#496184",
            )
            .configure_legend(labelColor="#dce9ff", titleColor="#dce9ff")
            .configure_view(strokeOpacity=0)
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
            .properties(height=300, padding={"left": 8, "right": 12, "top": 8, "bottom": 34})
            .configure_axis(
                labelColor="#dce9ff",
                titleColor="#dce9ff",
                labelPadding=8,
                titlePadding=10,
                gridColor="#30445f",
                tickColor="#496184",
                domainColor="#496184",
            )
            .configure_view(strokeOpacity=0)
            .interactive()
        )
        st.altair_chart(risk_chart, use_container_width=True)
        st.markdown("#### Recent Risk Signals")
        _render_sortable_centered_table(ktmp.tail(30), ["RiskScore", "RiskLevel", "RiskFlags"])

    st.markdown("#### Model Registry")
    try:
        import json

        with open("data/model_registry.json", "r", encoding="utf-8") as f:
            registry = json.load(f)
        generated_at = registry.get("generated_at")
        if generated_at:
            st.caption(f"Generated at: {generated_at}")

        models = registry.get("models", [])
        if isinstance(models, list) and models:
            model_df = pd.json_normalize(models, sep=".")
            preferred_cols = [
                "model_name",
                "model_version",
                "timestamp",
                "training_or_evaluation_window",
                "evaluation_summary.rows_scored",
                "evaluation_summary.quality_score_mean",
                "evaluation_summary.quality_score_median",
                "evaluation_summary.risk_score_mean",
                "evaluation_summary.regime_latest",
                "evaluation_summary.risk_level_latest",
            ]
            show_cols = [c for c in preferred_cols if c in model_df.columns]
            if not show_cols:
                show_cols = list(model_df.columns)
            _render_sortable_centered_table(model_df[show_cols], ["model_version", "evaluation_summary.rows_scored"])
        else:
            st.caption("No model entries found in model registry.")
    except Exception:
        st.caption("`data/model_registry.json` not available.")

    st.markdown("#### Model Health")
    try:
        import json

        with open("data/model_health_report.json", "r", encoding="utf-8") as f:
            health = json.load(f)
        generated_at = health.get("generated_at")
        if generated_at:
            st.caption(f"Generated at: {generated_at}")

        freshness = health.get("model_freshness", {})
        if isinstance(freshness, dict) and freshness:
            fresh_df = pd.DataFrame(
                [{"artifact": k, "status": str(v)} for k, v in freshness.items()]
            )
            status_l = fresh_df["status"].str.lower()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Fresh", int((status_l == "fresh").sum()))
            with c2:
                st.metric("Stale", int((status_l == "stale").sum()))
            with c3:
                st.metric("Other", int((~status_l.isin(["fresh", "stale"])).sum()))
            _render_sortable_centered_table(fresh_df, ["status"])

        coverage = health.get("input_cache_coverage", {})
        if isinstance(coverage, dict) and coverage:
            cov_rows: list[dict] = []
            for cache_name, payload in coverage.items():
                row = {"cache": cache_name}
                if isinstance(payload, dict):
                    row.update(payload)
                else:
                    row["value"] = payload
                cov_rows.append(row)
            cov_df = pd.DataFrame(cov_rows)
            if not cov_df.empty:
                preferred_cov_cols = ["cache", "exists", "schema_ok", "row_count", "status", "value"]
                cov_cols = [c for c in preferred_cov_cols if c in cov_df.columns]
                if not cov_cols:
                    cov_cols = list(cov_df.columns)
                _render_sortable_centered_table(cov_df[cov_cols], ["exists", "schema_ok"])
    except Exception:
        st.caption("`data/model_health_report.json` not available.")


def _show_explainability_tab() -> None:
    st.markdown("## Explainability and Evidence")
    st.caption("Artifacts are read from cache generated by the scheduled pipeline.")

    import json

    qexp, _qe = read_parquet_safe(QUALITY_EXPLAIN_PATH)
    qbase, _qb = read_parquet_safe(QUALITY_CACHE_PATH)
    revid, _re = read_parquet_safe(REGIME_EVIDENCE_PATH)
    revbase, _rb = read_parquet_safe(REGIME_CACHE_PATH)
    kevid, _ke = read_parquet_safe(RISK_EVIDENCE_PATH)
    kbase, _kb = read_parquet_safe(RISK_CACHE_PATH)
    fdf = _load_explainability_feature_union()

    # Section A: Entity Explainability
    st.markdown("### A. Entity Explainability")
    if qexp is None or qexp.empty:
        st.warning("Quality explanations cache missing. Showing fallback from quality scores cache when available.")
        if qbase is None or qbase.empty:
            st.caption("No quality score data available.")
        else:
            _render_explainability_table(qbase.head(50))
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
            _render_explainability_table(cdf)
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
                    _render_explainability_table(_format_evidence_card_table(pd.DataFrame(raw_vals)))

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
            _render_explainability_table(_format_evidence_points_table(edt))

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
            _render_explainability_table(_format_evidence_points_table(edt))

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


def _show_uncertainty_tab() -> None:
    st.markdown("## Uncertainty and Confidence")
    st.caption("Artifacts are read from cache generated by the scheduled pipeline.")

    qu, _qe = read_parquet_safe(QUALITY_UNCERTAINTY_PATH)
    rp, _re = read_parquet_safe(REGIME_PROB_PATH)
    ru, _ru = read_parquet_safe(RISK_UNCERTAINTY_PATH)
    qbase, _qb = read_parquet_safe(QUALITY_CACHE_PATH)
    rbase, _rb = read_parquet_safe(REGIME_CACHE_PATH)
    kbase, _kb = read_parquet_safe(RISK_CACHE_PATH)

    # Section A
    st.markdown("### A. QualityScore Uncertainty")
    src_q = qu if qu is not None and not qu.empty else qbase
    if src_q is None or src_q.empty:
        st.caption("Quality uncertainty cache missing and no quality score fallback available.")
    else:
        q = src_q.copy()
        q["Ticker"] = q["Ticker"].astype(str).str.upper().str.strip()
        tickers = sorted(q["Ticker"].dropna().unique().tolist())
        t = st.selectbox("Ticker", options=tickers, index=0, key="unc_ticker")
        row = q[q["Ticker"] == t].head(1).iloc[0]

        if {"ScoreP10", "ScoreP50", "ScoreP90", "TierMostLikely", "TierStability"}.issubset(set(q.columns)):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("ScoreP10", f"{float(row['ScoreP10']):.2f}")
            with c2:
                st.metric("ScoreP50", f"{float(row['ScoreP50']):.2f}")
            with c3:
                st.metric("ScoreP90", f"{float(row['ScoreP90']):.2f}")
            with c4:
                st.metric("TierStability", f"{float(row['TierStability']):.2f}")
            st.caption(
                f"TierMostLikely: {row.get('TierMostLikely', 'Unknown')}. "
                f"Tier stability is {float(row.get('TierStability', 0.0)):.2f}, "
                "which indicates how stable the tier is under feature noise."
            )
            show_chart = st.checkbox("Show uncertainty band chart", value=True, key="unc_show_q_chart")
            if show_chart:
                band_df = pd.DataFrame(
                    {
                        "Level": ["P10", "P50", "P90"],
                        "Score": [float(row["ScoreP10"]), float(row["ScoreP50"]), float(row["ScoreP90"])],
                    }
                )
                st.bar_chart(band_df.set_index("Level"))
        else:
            st.warning("Uncertainty cache missing. Showing point estimate only.")
            st.metric("QualityScore", f"{float(row.get('QualityScore', 0.0)):.2f}")

    # Section B
    st.markdown("### B. Regime Probabilities")
    src_r = rp if rp is not None and not rp.empty else rbase
    if src_r is None or src_r.empty:
        st.caption("Regime probabilities cache missing and no regime fallback available.")
    else:
        r = src_r.copy()
        r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
        r = r.dropna(subset=["Date"]).sort_values("Date")
        dates = [d.strftime("%Y-%m-%d") for d in r["Date"].tolist()]
        sel = st.selectbox("Regime probability date", options=dates, index=len(dates) - 1, key="unc_regime_date")
        row = r[r["Date"] == pd.to_datetime(sel)].tail(1).iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("RegimeLabel", str(row.get("RegimeLabel", "Unknown")))
        with c2:
            st.metric("ConfidenceScore", f"{float(row.get('ConfidenceScore', 0.0)):.2f}")

        if {"P_RiskOn", "P_Neutral", "P_RiskOff"}.issubset(set(r.columns)):
            prob_df = pd.DataFrame(
                [
                    {"Regime": "Risk On", "Probability": float(row["P_RiskOn"])},
                    {"Regime": "Neutral", "Probability": float(row["P_Neutral"])},
                    {"Regime": "Risk Off", "Probability": float(row["P_RiskOff"])},
                ]
            )
            _render_sortable_centered_table(prob_df, ["Probability"])
            st.bar_chart(prob_df.set_index("Regime"))
            if "RegimeStability_20d" in row:
                st.caption(f"RegimeStability_20d: {float(row['RegimeStability_20d']):.2f}")
        else:
            st.warning("Probability cache missing. Showing only point label and confidence.")

    # Section C
    st.markdown("### C. Risk Uncertainty")
    src_k = ru if ru is not None and not ru.empty else kbase
    if src_k is None or src_k.empty:
        st.caption("Risk uncertainty cache missing and no risk fallback available.")
    else:
        k = src_k.copy()
        k["Date"] = pd.to_datetime(k["Date"], errors="coerce")
        k = k.dropna(subset=["Date"]).sort_values("Date")
        dates = [d.strftime("%Y-%m-%d") for d in k["Date"].tolist()]
        sel = st.selectbox("Risk uncertainty date", options=dates, index=len(dates) - 1, key="unc_risk_date")
        lookback = st.slider("Last N days", min_value=30, max_value=365, value=60, step=10, key="unc_risk_lookback")
        row = k[k["Date"] == pd.to_datetime(sel)].tail(1).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("RiskScore", f"{float(row.get('RiskScore', 0.0)):.2f}")
        if {"RiskP10", "RiskP50", "RiskP90"}.issubset(set(k.columns)):
            with c2:
                st.metric("RiskP10", f"{float(row.get('RiskP10', 0.0)):.2f}")
            with c3:
                st.metric("RiskP50", f"{float(row.get('RiskP50', 0.0)):.2f}")
            with c4:
                st.metric("RiskP90", f"{float(row.get('RiskP90', 0.0)):.2f}")
            st.caption(
                f"RiskLevelMostLikely: {row.get('RiskLevelMostLikely', 'Unknown')} | "
                f"RiskLevelStability: {float(row.get('RiskLevelStability', 0.0)):.2f}"
            )
            tail = k.tail(int(lookback)).copy()
            if {"RiskP10", "RiskP90"}.issubset(set(tail.columns)):
                plot_df = tail[["Date", "RiskScore", "RiskP10", "RiskP90"]].dropna()
                line = alt.Chart(plot_df).mark_line(color="#ff9f43").encode(x="Date:T", y="RiskScore:Q")
                band = alt.Chart(plot_df).mark_area(opacity=0.2, color="#ff9f43").encode(
                    x="Date:T",
                    y="RiskP10:Q",
                    y2="RiskP90:Q",
                )
                st.altair_chart((band + line).properties(height=280), use_container_width=True)
        else:
            st.warning("Uncertainty cache missing. Showing only point estimate.")
            st.caption(f"RiskLevel: {row.get('RiskLevel', 'Unknown')}")


def _show_monitoring_tab() -> None:
    st.markdown("## Drift, Monitoring, and Early Warning")
    st.caption("Artifacts are read from cache generated by the scheduled pipeline.")

    import json

    drift_df, _de = read_parquet_safe(DRIFT_SIGNALS_PATH)
    drift_hist_df, _dh = read_parquet_safe(DRIFT_SIGNALS_HISTORY_PATH)
    alert_df, _ae = read_parquet_safe(ALERT_LOG_PATH)
    regime_df, _re = read_parquet_safe(REGIME_CACHE_PATH)
    risk_df, _rk = read_parquet_safe(RISK_CACHE_PATH)

    if drift_df is None or drift_df.empty:
        st.warning("Monitoring artifacts missing. Run pipeline with run_monitoring=True.")
        return

    def _filter_by_date(df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
        if df is None or df.empty or "Date" not in df.columns:
            return df
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        out = out.dropna(subset=["Date"])
        return out[(out["Date"] >= start) & (out["Date"] <= end)].copy()

    date_chunks: list[pd.Series] = []
    for src in [drift_hist_df, drift_df, alert_df, regime_df, risk_df]:
        if src is None or src.empty or "Date" not in src.columns:
            continue
        s = pd.to_datetime(src["Date"], errors="coerce", utc=True).dt.tz_localize(None).dropna()
        if not s.empty:
            date_chunks.append(s)

    if date_chunks:
        all_dates = pd.concat(date_chunks, axis=0)
        min_date = pd.Timestamp(all_dates.min()).normalize()
        max_date = pd.Timestamp(all_dates.max()).normalize()
    else:
        max_date = pd.Timestamp.utcnow().normalize()
        min_date = max_date - pd.Timedelta(days=365)

    mon_range = st.selectbox(
        "Date range",
        ["Last 30D", "Last 60D", "Last 90D", "Last 180D", "All"],
        index=2,
        key="mon_date_range_preset",
    )
    if mon_range == "All":
        win_start = min_date
        win_end = max_date
    else:
        days = int(mon_range.split()[1].replace("D", ""))
        win_start = max(min_date, max_date - pd.Timedelta(days=days))
        win_end = max_date
    if win_start > win_end:
        win_start, win_end = win_end, win_start
    win_end = win_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    drift_df_f = _filter_by_date(drift_df, win_start, win_end)
    drift_hist_df_f = _filter_by_date(drift_hist_df, win_start, win_end)
    alert_df_f = _filter_by_date(alert_df, win_start, win_end)
    regime_df_f = _filter_by_date(regime_df, win_start, win_end)
    risk_df_f = _filter_by_date(risk_df, win_start, win_end)

    # Section A: summary
    st.markdown("### A. Monitoring Summary")
    worst = "Stable"
    src_summary = drift_df_f if (drift_df_f is not None and not drift_df_f.empty) else drift_df
    if (src_summary["DriftLevel"] == "Severe").any():
        worst = "Severe"
    elif (src_summary["DriftLevel"] == "Drift").any():
        worst = "Drift"
    signal_unstable = "Unstable" if ((src_summary["MetricType"] == "Signal") & (src_summary["DriftLevel"] != "Stable")).any() else "Stable"
    src_alerts = alert_df_f if (alert_df_f is not None and not alert_df_f.empty) else alert_df
    crit = int(((src_alerts is not None) and (not src_alerts.empty) and (src_alerts["Severity"] == "Critical").sum()) or 0)
    warn = int(((src_alerts is not None) and (not src_alerts.empty) and (src_alerts["Severity"] == "Warning").sum()) or 0)
    info = int(((src_alerts is not None) and (not src_alerts.empty) and (src_alerts["Severity"] == "Info").sum()) or 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Feature Drift", worst)
    with c2:
        st.metric("Signal Stability", signal_unstable)
    with c3:
        st.metric("Active Alerts", f"C:{crit} W:{warn} I:{info}")

    # Section B: alerts
    st.markdown("### B. Active Alerts")
    if src_alerts is None or src_alerts.empty:
        st.caption("No alerts available.")
    else:
        a = src_alerts.copy()
        a["Date"] = pd.to_datetime(a["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        recent = a.sort_values("Date", ascending=False).head(50)
        cols = [c for c in ["Severity", "AlertType", "Title", "Date"] if c in recent.columns]
        _render_sortable_all_but_first_table(recent[cols])

        choices = [f"{r.Severity} | {r.AlertType} | {r.Date:%Y-%m-%d}" for r in recent.itertuples()]
        idx = st.selectbox("Select alert", options=list(range(len(choices))), format_func=lambda i: choices[i], key="mon_alert_pick")
        row = recent.iloc[int(idx)]
        st.caption(str(row.get("Description", "")))
        try:
            ev = json.loads(str(row.get("EvidenceJSON", "{}")))
            if isinstance(ev, dict) and ev:
                ev_df = pd.json_normalize(ev, sep=".")
                _render_sortable_all_but_first_table(ev_df)
            elif isinstance(ev, list) and ev:
                _render_sortable_all_but_first_table(pd.json_normalize(ev, sep="."))
            else:
                st.caption("No structured evidence details available.")
        except Exception:
            st.caption("Evidence JSON unavailable.")
        st.caption(f"Suggested action: {row.get('SuggestedAction', 'N/A')}")

    # Section C: drift dashboard
    st.markdown("### C. Drift Dashboard")
    levels = ["All", "Stable", "Drift", "Severe"]
    types = ["All", "Feature", "Signal"]
    c1, c2 = st.columns(2)
    with c1:
        level_pick = st.selectbox("DriftLevel filter", options=levels, index=0, key="mon_level_filter")
    with c2:
        type_pick = st.selectbox("MetricType filter", options=types, index=0, key="mon_type_filter")
    d = src_summary.copy()
    if level_pick != "All":
        d = d[d["DriftLevel"] == level_pick]
    if type_pick != "All":
        d = d[d["MetricType"] == type_pick]
    d = d.sort_values("DriftScore", ascending=False)
    _render_sortable_all_but_first_table(d)

    trend_src = drift_hist_df_f if (drift_hist_df_f is not None and not drift_hist_df_f.empty) else src_summary
    if trend_src is not None and not trend_src.empty:
        metric_names = sorted(trend_src["MetricName"].astype(str).unique().tolist())
        sel_metric = st.selectbox("Drift metric trend", options=metric_names, index=0, key="mon_metric_trend")
        trend = trend_src[trend_src["MetricName"] == sel_metric].copy()
        if not trend.empty:
            trend["Date"] = pd.to_datetime(trend["Date"], errors="coerce")
            trend["DriftScore"] = pd.to_numeric(trend["DriftScore"], errors="coerce")
            trend = trend.dropna(subset=["Date", "DriftScore"]).sort_values("Date")
            if not trend.empty:
                trend_chart = (
                    alt.Chart(trend)
                    .mark_line(color="#7ec8ff")
                    .encode(
                        x=alt.X("Date:T", scale=alt.Scale(domain=[win_start, win_end]), title=""),
                        y=alt.Y("DriftScore:Q", title=""),
                    )
                    .properties(height=300)
                    .interactive()
                )
                st.altair_chart(trend_chart, use_container_width=True)

    # Section D: signal stability
    st.markdown("### D. Signal Stability")
    c1, c2, c3 = st.columns(3)
    with c1:
        if regime_df_f is not None and not regime_df_f.empty and "RegimeLabel" in regime_df_f.columns:
            rr = regime_df_f.copy()
            rr["Date"] = pd.to_datetime(rr["Date"], errors="coerce")
            rr = rr.dropna(subset=["Date"]).sort_values("Date").tail(60)
            flip = float((rr["RegimeLabel"] != rr["RegimeLabel"].shift(1)).mean()) if len(rr) > 1 else 0.0
            avg_conf = float(pd.to_numeric(rr.get("ConfidenceScore"), errors="coerce").mean()) if "ConfidenceScore" in rr.columns else float("nan")
            st.metric("Regime Flip Rate (60d)", f"{flip:.2f}")
            if pd.notna(avg_conf):
                st.caption(f"Avg confidence (60d): {avg_conf:.2f}")
    with c2:
        if risk_df_f is not None and not risk_df_f.empty and "RiskScore" in risk_df_f.columns:
            rk = risk_df_f.copy()
            rk["Date"] = pd.to_datetime(rk["Date"], errors="coerce")
            rk["RiskScore"] = pd.to_numeric(rk["RiskScore"], errors="coerce")
            rk = rk.dropna(subset=["Date", "RiskScore"]).sort_values("Date").tail(60)
            vol = float(rk["RiskScore"].std(ddof=1)) if len(rk) > 1 else 0.0
            chg = int((rk["RiskLevel"] != rk["RiskLevel"].shift(1)).sum()) if "RiskLevel" in rk.columns else 0
            st.metric("RiskScore Volatility (60d)", f"{vol:.2f}")
            st.caption(f"Risk level changes (60d): {chg}")
    with c3:
        q_proxy = src_summary[src_summary["MetricName"] == "QualityProxyDrift"]
        if not q_proxy.empty:
            val = float(q_proxy["DriftScore"].iloc[-1])
            st.metric("Quality Instability Proxy", f"{val:.3f}")

    # Optional monitoring reports
    st.markdown("#### Monitoring Reports")
    try:
        with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
            dr = json.load(f)
        st.markdown("**Drift Report**")
        if isinstance(dr, dict) and dr:
            created_at = dr.get("created_at") or dr.get("generated_at")
            if created_at:
                st.caption(f"Generated at: {created_at}")

            top_features = dr.get("top_drifting_features", [])
            top_signals = dr.get("top_drifting_signals", [])
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Top Feature Drivers", len(top_features) if isinstance(top_features, list) else 0)
            with c2:
                st.metric("Top Signal Drivers", len(top_signals) if isinstance(top_signals, list) else 0)
            with c3:
                worst_level = "Unknown"
                if isinstance(top_features, list) and top_features:
                    levels = pd.Series([str(x.get("DriftLevel", "")) for x in top_features]).str.lower()
                    if (levels == "severe").any():
                        worst_level = "Severe"
                    elif (levels == "drift").any():
                        worst_level = "Drift"
                    elif (levels == "stable").any():
                        worst_level = "Stable"
                st.metric("Worst Feature Drift", worst_level)

            window_settings = dr.get("window_settings_used", {})
            if isinstance(window_settings, dict) and window_settings:
                st.markdown("Window Settings")
                _render_sortable_all_but_first_table(pd.DataFrame([window_settings]))

            if isinstance(top_features, list) and top_features:
                st.markdown("Top Drifting Features")
                tf_df = _filter_by_date(pd.DataFrame(top_features), win_start, win_end)
                _render_sortable_all_but_first_table(tf_df if tf_df is not None else pd.DataFrame(top_features))
            if isinstance(top_signals, list) and top_signals:
                st.markdown("Top Drifting Signals")
                ts_df = _filter_by_date(pd.DataFrame(top_signals), win_start, win_end)
                _render_sortable_all_but_first_table(ts_df if ts_df is not None else pd.DataFrame(top_signals))

            coverage = dr.get("data_coverage_stats", {})
            if isinstance(coverage, dict) and coverage:
                st.markdown("Coverage Summary")
                cov_simple = {
                    k: v for k, v in coverage.items() if k != "missing_counts"
                }
                if cov_simple:
                    _render_sortable_first_col_centered_table(pd.DataFrame([cov_simple]))
                missing_counts = coverage.get("missing_counts", {})
                if isinstance(missing_counts, dict) and missing_counts:
                    miss_df = pd.DataFrame(
                        [{"MetricName": k, "MissingCount": int(v)} for k, v in missing_counts.items()]
                    ).sort_values("MissingCount", ascending=False)
                    st.markdown("Missing Feature Counts")
                    _render_sortable_all_but_first_table(miss_df)

            notes = dr.get("warnings_and_fallbacks_used", [])
            if isinstance(notes, list) and notes:
                st.markdown("Warnings and Fallbacks")
                for n in notes:
                    st.caption(f"- {n}")

            summary = dr.get("short_narrative_summary")
            if summary:
                st.caption(str(summary))
        elif isinstance(dr, list) and dr:
            dr_df = _filter_by_date(pd.json_normalize(dr, sep="."), win_start, win_end)
            _render_sortable_all_but_first_table(dr_df if dr_df is not None else pd.json_normalize(dr, sep="."))
        else:
            st.caption("Drift report is empty.")
    except Exception:
        st.caption("drift_report.json not available.")
    try:
        with open(MONITORING_HEALTH_PATH, "r", encoding="utf-8") as f:
            mh = json.load(f)
        st.markdown("**Monitoring Health Report**")
        if isinstance(mh, dict) and mh:
            created_at = mh.get("created_at") or mh.get("generated_at")
            if created_at:
                st.caption(f"Generated at: {created_at}")

            freshness = mh.get("artifact_freshness", {})
            status_l = pd.Series(dtype=str)
            if isinstance(freshness, dict) and freshness:
                status_l = pd.Series([str(v) for v in freshness.values()]).str.lower()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Alerts Generated", int(mh.get("alerts_generated_count") or 0))
            with c2:
                st.metric("Worst Drift Level", str(mh.get("worst_drift_level", "Unknown")))
            with c3:
                st.metric("Fresh Artifacts", int((status_l == "fresh").sum()) if not status_l.empty else 0)

            if isinstance(freshness, dict) and freshness:
                fresh_df = pd.DataFrame(
                    [{"artifact": k, "status": str(v)} for k, v in freshness.items()]
                )
                st.markdown("Artifact Freshness")
                _render_sortable_all_but_first_table(fresh_df)

            coverage = mh.get("coverage", {})
            if isinstance(coverage, dict) and coverage:
                st.markdown("Coverage Summary")
                cov_simple = {
                    k: v for k, v in coverage.items() if k != "missing_counts"
                }
                if cov_simple:
                    _render_sortable_first_col_centered_table(pd.DataFrame([cov_simple]))
                missing_counts = coverage.get("missing_counts", {})
                if isinstance(missing_counts, dict) and missing_counts:
                    miss_df = pd.DataFrame(
                        [{"MetricName": k, "MissingCount": int(v)} for k, v in missing_counts.items()]
                    ).sort_values("MissingCount", ascending=False)
                    st.markdown("Coverage Missing Counts")
                    _render_sortable_all_but_first_table(miss_df)

            mfc = mh.get("missing_feature_counts", {})
            if isinstance(mfc, dict) and mfc:
                mfc_df = pd.DataFrame(
                    [{"MetricName": k, "MissingCount": int(v)} for k, v in mfc.items()]
                ).sort_values("MissingCount", ascending=False)
                st.markdown("Missing Feature Counts")
                _render_sortable_all_but_first_table(mfc_df)

            notes = mh.get("runtime_notes", [])
            if isinstance(notes, list) and notes:
                st.markdown("Runtime Notes")
                for n in notes:
                    st.caption(f"- {n}")
        elif isinstance(mh, list) and mh:
            mh_df = _filter_by_date(pd.json_normalize(mh, sep="."), win_start, win_end)
            _render_sortable_all_but_first_table(mh_df if mh_df is not None else pd.json_normalize(mh, sep="."))
        else:
            st.caption("Monitoring health report is empty.")
    except Exception:
        st.caption("monitoring_health_report.json not available.")


_show_signal_banner()

stock_tab, fi_tab, sim_tab, di_tab, ex_tab, un_tab, mon_tab = st.tabs(
    [
        "Stock Screener",
        "Bond & Treasury Screener",
        "Portfolio Decision Simulator",
        "Decision Intelligence",
        "Explainability and Evidence",
        "Uncertainty and Confidence",
        "Drift, Monitoring, and Early Warning",
    ]
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
with un_tab:
    _show_uncertainty_tab()
with mon_tab:
    _show_monitoring_tab()

