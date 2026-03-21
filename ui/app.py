from __future__ import annotations

from dataclasses import dataclass
import html
import sys
import textwrap
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    from st_aggrid.shared import JsCode
    try:
        from st_aggrid import GridUpdateMode
    except Exception:
        GridUpdateMode = None
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    JsCode = None
    GridUpdateMode = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"

from data_pipeline.cache_manager import get_cache_status, read_parquet_safe
from data_pipeline.data_fetcher import FI_SCHEMA_COLUMNS, SCHEMA_COLUMNS
from reports.decision_brief import generate_decision_brief
from simulation.portfolio_simulator import simulate_portfolio

st.set_page_config(page_title="Investment Lab", layout="wide")

STOCK_UNIVERSE_OPTIONS = ["S&P 500", "Nasdaq 100"]
FI_UNIVERSE_OPTIONS = ["US Treasuries", "Bond ETFs"]
SIM_BENCHMARK_OPTIONS = ["SPY", "QQQ", "DIA", "IWM"]
MAX_AGE_DAYS = 7
MIN_REFRESH_SUCCESS_RATIO = 0.25
PRICE_SCHEMA_COLUMNS = ["Ticker", "Date", "AdjClose", "Close", "Volume"]
PRICES_CACHE_PATH = str(DATA_DIR / "prices_cache.parquet")
PRICES_HEALTH_PATH = str(DATA_DIR / "prices_health_report.json")
QUALITY_CACHE_PATH = str(DATA_DIR / "quality_scores_cache.parquet")
REGIME_CACHE_PATH = str(DATA_DIR / "regime_cache.parquet")
RISK_CACHE_PATH = str(DATA_DIR / "risk_signals_cache.parquet")
QUALITY_EXPLAIN_PATH = str(DATA_DIR / "quality_explanations_cache.parquet")
REGIME_EVIDENCE_PATH = str(DATA_DIR / "regime_evidence_cache.parquet")
RISK_EVIDENCE_PATH = str(DATA_DIR / "risk_evidence_cache.parquet")
QUALITY_UNCERTAINTY_PATH = str(DATA_DIR / "quality_uncertainty_cache.parquet")
REGIME_PROB_PATH = str(DATA_DIR / "regime_probabilities_cache.parquet")
RISK_UNCERTAINTY_PATH = str(DATA_DIR / "risk_uncertainty_cache.parquet")
DRIFT_SIGNALS_PATH = str(DATA_DIR / "drift_signals_cache.parquet")
DRIFT_SIGNALS_HISTORY_PATH = str(DATA_DIR / "drift_signals_history.parquet")
DRIFT_REPORT_PATH = str(DATA_DIR / "drift_report.json")
ALERT_LOG_PATH = str(DATA_DIR / "alert_log.parquet")
MONITORING_HEALTH_PATH = str(DATA_DIR / "monitoring_health_report.json")
FUNDAMENTALS_CACHE_PATH = str(DATA_DIR / "fundamentals_cache.parquet")
FUNDAMENTALS_CACHE_SP500_PATH = str(DATA_DIR / "fundamentals_cache_sp500.parquet")
FUNDAMENTALS_CACHE_NASDAQ100_PATH = str(DATA_DIR / "fundamentals_cache_nasdaq100.parquet")
FUNDAMENTALS_HEALTH_PATH = str(DATA_DIR / "fundamentals_health_report.json")
FUNDAMENTALS_HEALTH_SP500_PATH = str(DATA_DIR / "fundamentals_health_report_sp500.json")
FUNDAMENTALS_HEALTH_NASDAQ100_PATH = str(DATA_DIR / "fundamentals_health_report_nasdaq100.json")
FIXED_INCOME_CACHE_TREASURY_PATH = str(DATA_DIR / "fixed_income_cache_treasury.parquet")
FIXED_INCOME_CACHE_BOND_ETF_PATH = str(DATA_DIR / "fixed_income_cache_bond_etf.parquet")
FIXED_INCOME_HEALTH_TREASURY_PATH = str(DATA_DIR / "fixed_income_health_treasury.json")
FIXED_INCOME_HEALTH_BOND_ETF_PATH = str(DATA_DIR / "fixed_income_health_bond_etf.json")
TREASURY_YIELDS_CACHE_PATH = str(DATA_DIR / "treasury_yields_cache.parquet")
MODEL_REGISTRY_PATH = str(DATA_DIR / "model_registry.json")
MODEL_HEALTH_PATH = str(DATA_DIR / "model_health_report.json")
RUN_ARTIFACTS_DIR = str(DATA_DIR / "run_artifacts")


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
        .ii-insights-hdr {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }
        .ii-insights-hdr h4 {
            margin: 0;
        }
        .ii-insights-info {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text);
            background: var(--surface-2);
            border: 1px solid var(--border);
            cursor: help;
            user-select: none;
            position: relative;
        }
        .ii-insights-info:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 0;
            bottom: calc(100% + 10px);
            width: min(430px, calc(100vw - 36px));
            background: #0a1324;
            color: #e8f1ff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 0.82rem;
            line-height: 1.35;
            font-weight: 500;
            text-align: left;
            white-space: normal;
            z-index: 9999;
            pointer-events: none;
            box-shadow: var(--shadow);
        }

        .ii-insights ul {
            margin: 0;
            padding-left: 0;
            list-style: none;
        }
        .ii-insights-cols {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            column-gap: 24px;
            row-gap: 0;
        }

        .ii-insights li {
            margin: 1px 0;
            line-height: 1.2;
            color: #dce9ff;
        }
        .ii-insights li::before {
            content: "\\2022";
            display: inline-block;
            margin-right: 10px;
            color: #cfe2ff;
        }

        .diw-wrap {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(12, 27, 50, 0.96), rgba(12, 27, 50, 0.90));
            padding: 14px 16px;
            box-shadow: var(--shadow);
            margin-bottom: 12px;
        }
        .diw-kicker {
            color: #95aacd;
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
            margin-bottom: 2px;
        }
        .diw-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 10px;
        }
        .diw-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            color: #e8f1ff;
        }
        .diw-sub {
            color: #9db3d6;
            font-size: 1.05rem;
            font-weight: 600;
        }
        .diw-live {
            color: #9bd8cb;
            font-size: 1rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .diw-cards {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 4px;
        }
        .diw-card {
            border: 1px solid rgba(135, 162, 196, 0.26);
            border-radius: 14px;
            background: rgba(10, 22, 41, 0.55);
            padding: 12px 14px;
            position: relative;
            min-height: 118px;
        }
        .diw-card-blue::before,
        .diw-card-amber::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            right: 0;
            height: 3px;
            border-top-left-radius: 14px;
            border-top-right-radius: 14px;
            background: linear-gradient(90deg, #2ec5ff, #1e8dff);
        }
        .diw-card-amber::before {
            background: linear-gradient(90deg, #ffb549, #ff9f43);
        }
        .diw-card-k {
            color: #9fb5d7;
            font-size: 0.98rem;
            margin-bottom: 2px;
        }
        .diw-card-v {
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.1;
            color: #f1f6ff;
        }
        .diw-card-note {
            color: #c6d8f3;
            font-size: 0.95rem;
            margin-top: 2px;
        }
        .diw-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-top: 8px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #d9e8ff;
            border: 1px solid rgba(80, 122, 170, 0.45);
            background: rgba(33, 68, 112, 0.42);
        }
        .diw-chip-risk {
            color: #ffd89d;
            border-color: rgba(255, 181, 73, 0.5);
            background: rgba(255, 166, 52, 0.18);
        }
        .diw-section-title {
            color: #95aacd;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 16px 0 8px 0;
        }
        .diw-breakdown {
            border: 1px solid rgba(135, 162, 196, 0.26);
            border-radius: 14px;
            background: rgba(10, 22, 41, 0.55);
            padding: 12px 14px;
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 8px 14px;
            align-items: start;
        }
        .diw-big {
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .diw-muted {
            color: #9db3d6;
            font-size: 0.95rem;
        }
        .diw-rows {
            margin-top: 8px;
        }
        .diw-row {
            display: grid;
            grid-template-columns: 86px 1fr 40px 42px;
            gap: 10px;
            align-items: center;
            margin-bottom: 7px;
        }
        .diw-name-strong { color: #35d39f; font-weight: 700; }
        .diw-name-neutral { color: #63b1ff; font-weight: 700; }
        .diw-name-weak { color: #ff6a6a; font-weight: 700; }
        .diw-track {
            height: 8px;
            background: rgba(132, 154, 184, 0.26);
            border-radius: 999px;
            overflow: hidden;
        }
        .diw-fill-strong { height: 100%; background: linear-gradient(90deg, #2ed39f, #1dbf8f); }
        .diw-fill-neutral { height: 100%; background: linear-gradient(90deg, #65b6ff, #3f99f3); }
        .diw-fill-weak { height: 100%; background: linear-gradient(90deg, #ff7676, #ff5252); }
        .diw-right-num {
            text-align: right;
            color: #d9e6fb;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }
        .diw-changed {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .diw-change-card {
            border: 1px solid rgba(135, 162, 196, 0.26);
            border-radius: 14px;
            background: rgba(10, 22, 41, 0.55);
            padding: 12px 14px;
            min-height: 150px;
        }
        .diw-change-icon {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            margin-bottom: 8px;
            color: #e8f1ff;
            border: 1px solid rgba(155, 177, 206, 0.35);
        }
        .diw-icon-blue { background: rgba(62, 122, 196, 0.42); }
        .diw-icon-amber { background: rgba(255, 166, 52, 0.24); color: #ffd89d; }
        .diw-change-title {
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 4px;
        }
        .diw-change-copy {
            color: #d4e2f9;
            font-size: 1.05rem;
            line-height: 1.35;
            min-height: 58px;
        }
        .diw-change-foot {
            margin-top: 8px;
            color: #9fb5d7;
            font-size: 0.98rem;
            font-weight: 700;
        }
        .diw-up {
            color: #ff9aa0;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .diw-approach {
            border: 1px solid rgba(135, 162, 196, 0.26);
            border-radius: 14px;
            background: rgba(10, 22, 41, 0.55);
            padding: 12px 14px;
        }
        .diw-item {
            display: grid;
            grid-template-columns: 38px 1fr;
            gap: 10px;
            align-items: start;
            padding: 9px 0;
            border-bottom: 1px solid rgba(135, 162, 196, 0.20);
        }
        .diw-item:last-child {
            border-bottom: none;
        }
        .diw-item-icon {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            border: 1px solid rgba(150, 173, 204, 0.32);
        }
        .diw-icon-a { background: rgba(56, 110, 188, 0.42); color: #d4e5ff; }
        .diw-icon-b { background: rgba(177, 141, 23, 0.30); color: #fff0c2; }
        .diw-icon-c { background: rgba(48, 61, 52, 0.55); color: #bcf1bc; }
        .diw-item-title {
            font-size: 1.9rem;
            line-height: 1.1;
            font-weight: 800;
            margin-bottom: 2px;
            color: #ecf3ff;
        }
        .diw-item-copy {
            color: #b7cae8;
            font-size: 1.02rem;
            line-height: 1.3;
        }
        .diw-status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 8px 14px;
            align-items: center;
            border: 1px solid rgba(135, 162, 196, 0.26);
            border-radius: 12px;
            background: rgba(10, 22, 41, 0.55);
            padding: 10px 12px;
            margin-bottom: 10px;
            color: #d9e7ff;
            font-size: 0.97rem;
        }
        .diw-sep {
            color: #90a9ce;
            opacity: 0.8;
        }
        .diw-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid rgba(133, 163, 202, 0.45);
            background: rgba(44, 77, 120, 0.32);
            font-weight: 700;
            color: #d8e7ff;
            font-size: 0.92rem;
        }
        .diw-status-pill-high {
            border-color: rgba(255, 134, 134, 0.62);
            background: rgba(153, 51, 51, 0.35);
            color: #ffd5d5;
        }
        .diw-status-pill-medium {
            border-color: rgba(255, 189, 102, 0.62);
            background: rgba(145, 92, 35, 0.32);
            color: #ffe2b5;
        }
        .diw-status-pill-low {
            border-color: rgba(102, 214, 169, 0.52);
            background: rgba(29, 113, 84, 0.28);
            color: #c6f3e2;
        }
        .diw-action-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .diw-action-card {
            border: 1px solid rgba(135, 162, 196, 0.28);
            border-radius: 14px;
            background: rgba(10, 22, 41, 0.55);
            padding: 12px 14px;
        }
        .diw-action-title {
            color: #edf4ff;
            font-weight: 800;
            font-size: 1.18rem;
            line-height: 1.2;
            margin-bottom: 4px;
        }
        .diw-action-copy {
            color: #c7d9f5;
            font-size: 1.0rem;
            line-height: 1.35;
            min-height: 54px;
        }
        .diw-action-meta {
            margin-top: 8px;
            color: #9fb6da;
            font-size: 0.95rem;
        }
        .diw-action-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-top: 10px;
            border: 1px solid rgba(132, 161, 198, 0.45);
            border-radius: 10px;
            padding: 7px 12px;
            color: #e8f1ff;
            font-weight: 700;
            text-decoration: none;
            background: rgba(40, 65, 100, 0.35);
            font-size: 0.96rem;
        }
        @media (max-width: 980px) {
            .diw-cards, .diw-changed, .diw-action-grid {
                grid-template-columns: 1fr;
            }
            .diw-breakdown {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_premium_theme()


def _stock_paths(universe: str) -> tuple[str, str]:
    key = (universe or "").strip().lower().replace("&", "and").replace(" ", "").replace("-", "")
    if key in {"sandp500", "sp500"}:
        return FUNDAMENTALS_CACHE_SP500_PATH, FUNDAMENTALS_HEALTH_SP500_PATH
    if key in {"nasdaq100", "ndx"}:
        return FUNDAMENTALS_CACHE_NASDAQ100_PATH, FUNDAMENTALS_HEALTH_NASDAQ100_PATH
    raise ValueError(f"Unsupported stock universe: {universe}")


def _fi_paths(universe: str) -> tuple[str, str]:
    key = (universe or "").strip().lower().replace(" ", "").replace("-", "")
    if key in {"ustreasuries", "treasury", "treasuries"}:
        return FIXED_INCOME_CACHE_TREASURY_PATH, FIXED_INCOME_HEALTH_TREASURY_PATH
    if key in {"bondetfs", "bondetf", "etf"}:
        return FIXED_INCOME_CACHE_BOND_ETF_PATH, FIXED_INCOME_HEALTH_BOND_ETF_PATH
    raise ValueError(f"Unsupported fixed-income universe: {universe}")


@st.cache_data(show_spinner=False, ttl=300)
def _load_fundamentals_union() -> pd.DataFrame:
    paths = [
        FUNDAMENTALS_CACHE_PATH,
        FUNDAMENTALS_CACHE_SP500_PATH,
        FUNDAMENTALS_CACHE_NASDAQ100_PATH,
    ]
    keep_priority_cols = [
        "Ticker",
        "Company",
        "Sector",
        "Close",
        "MarketCap",
        "EBITDA_Margin",
        "ROE",
        "Revenue_Growth_YoY_Pct",
        "Earnings_Growth_Pct",
        "PE_Ratio",
        "PEG_Ratio",
        "Rule_of_40",
    ]
    frames: list[pd.DataFrame] = []
    for p in paths:
        df, _err = read_parquet_safe(p)
        if df is None or df.empty:
            continue
        if "Ticker" not in df.columns:
            continue
        keep_cols = [c for c in keep_priority_cols if c in df.columns]
        frames.append(df[keep_cols].copy())
    if not frames:
        return pd.DataFrame(columns=["Ticker", "Company", "Sector", "MarketCap"])
    out = pd.concat(frames, axis=0, ignore_index=True)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    if "MarketCap" in out.columns:
        out["MarketCap"] = pd.to_numeric(out.get("MarketCap"), errors="coerce")
    out = out.dropna(subset=["Ticker"])

    def _first_non_null(s: pd.Series):
        t = s.dropna()
        if t.empty:
            return np.nan
        txt = t.astype(str).str.strip()
        txt = txt[txt != ""]
        return txt.iloc[0] if not txt.empty else np.nan

    agg: dict[str, str | callable] = {}
    for col in out.columns:
        if col == "Ticker":
            continue
        if col == "MarketCap":
            agg[col] = "max"
        else:
            agg[col] = _first_non_null
    out = out.groupby("Ticker", as_index=False).agg(agg)
    return out


@st.cache_data(show_spinner=False, ttl=300)
def _build_portfolio_suggestion_base() -> pd.DataFrame:
    qdf, _qe = read_parquet_safe(QUALITY_CACHE_PATH)
    if qdf is None or qdf.empty:
        return pd.DataFrame()
    q = qdf.copy()
    q["Ticker"] = q.get("Ticker", "").astype(str).str.upper().str.strip()
    q["QualityScore"] = pd.to_numeric(q.get("QualityScore"), errors="coerce")
    q["QualityTier"] = q.get("QualityTier", "").astype(str)
    q = q.dropna(subset=["Ticker", "QualityScore"])
    q = q[q["Ticker"] != ""]

    fdf = _load_fundamentals_union()
    if fdf is None or fdf.empty:
        return q
    f = fdf.copy()
    f["Ticker"] = f["Ticker"].astype(str).str.upper().str.strip()
    return q.merge(f, on="Ticker", how="left")


def _load_explainability_feature_union() -> pd.DataFrame:
    paths = [
        FUNDAMENTALS_CACHE_PATH,
        FUNDAMENTALS_CACHE_SP500_PATH,
        FUNDAMENTALS_CACHE_NASDAQ100_PATH,
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


def _default_sim_holdings_rows() -> list[dict[str, object]]:
    return [
        {"Ticker": "AAPL", "WeightPct": 25.0},
        {"Ticker": "MSFT", "WeightPct": 25.0},
        {"Ticker": "NVDA", "WeightPct": 25.0},
        {"Ticker": "GOOGL", "WeightPct": 25.0},
    ]


def _clean_sim_holdings_rows(rows: list[dict[str, object]] | None) -> list[dict[str, object]]:
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in rows or []:
        ticker = str((row or {}).get("Ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        weight = pd.to_numeric(pd.Series([(row or {}).get("WeightPct")]), errors="coerce").iloc[0]
        weight_pct = 0.0 if pd.isna(weight) else float(weight)
        out.append({"Ticker": ticker, "WeightPct": weight_pct})
    return out


def _sim_rows_to_inputs(rows: list[dict[str, object]]) -> tuple[str, str, float]:
    clean = _clean_sim_holdings_rows(rows)
    tickers = [str(r["Ticker"]) for r in clean]
    weights_pct = [float(r["WeightPct"]) for r in clean]
    ticker_text = ", ".join(tickers)
    manual_weights_text = ", ".join([f"{(w / 100.0):.6f}" for w in weights_pct])
    total_pct = float(sum(weights_pct))
    return ticker_text, manual_weights_text, total_pct


def _auto_risk_free_rate_pct(*, treasury_path: str = TREASURY_YIELDS_CACHE_PATH) -> tuple[float, str]:
    fallback = 4.0
    candidate_paths: list[str] = [treasury_path]
    tpath = Path(treasury_path)
    if not tpath.is_absolute():
        candidate_paths.append(str((ROOT / tpath).resolve()))

    tdf = None
    _err = None
    for p in candidate_paths:
        tdf, _err = read_parquet_safe(p)
        if tdf is not None and not tdf.empty:
            break
    if tdf is None or tdf.empty:
        return fallback, "Fallback default (treasury cache unavailable)"

    cols = list(tdf.columns)
    canon = {str(c).lower().replace("_", "").replace(" ", ""): c for c in cols}

    def _pick(candidates: list[str]) -> str | None:
        for c in candidates:
            key = c.lower().replace("_", "").replace(" ", "")
            if key in canon:
                return canon[key]
        return None

    date_col = _pick(["Date", "AsOfDate", "Timestamp"])
    y3m_col = _pick(["3M", "DGS3MO", "Yield3M", "UST3M"])
    y2_col = _pick(["2Y", "DGS2", "Yield2Y", "UST2Y"])
    y10_col = _pick(["10Y", "DGS10", "Yield10Y", "UST10Y"])
    chosen_col = y3m_col or y2_col or y10_col
    chosen_lbl = "3M" if chosen_col == y3m_col else ("2Y" if chosen_col == y2_col else ("10Y" if chosen_col == y10_col else "N/A"))
    if chosen_col is None:
        return fallback, "Fallback default (no treasury yield column found)"

    tmp = tdf.copy()
    if date_col is not None:
        tmp["_date"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.sort_values("_date")
    tmp["_yield"] = pd.to_numeric(tmp[chosen_col], errors="coerce")
    ser = tmp["_yield"].dropna()
    if ser.empty:
        return fallback, f"Fallback default ({chosen_lbl} yield unavailable)"

    val = float(ser.iloc[-1])
    if 0.0 <= val <= 1.0:
        val *= 100.0
    if not (-5.0 <= val <= 25.0):
        return fallback, f"Fallback default ({chosen_lbl} yield out of range)"
    return val, f"Auto from treasury {chosen_lbl} yield"


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
        pagination_footer_h = 44 if (page_size is not None and int(page_size) > 0) else 0
        grid_height = int(header_h + (max(1, visible_rows) * row_h) + pagination_footer_h + 2)
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
                    <span class="stock-detail-badge-info stock-detail-tip stock-detail-tip-right stock-detail-tip-below" data-tooltip="Quality Score is a 0-100 weighted composite of percentile-ranked inputs: Revenue Growth (20%), EBITDA Margin (20%), ROE (20%), Free Cash Flow Margin (20%), Volatility Stability (10%), and Drawdown Stability (10%). Missing values are assigned a neutral percentile of 0.5. Tiers: Strong (>=67), Neutral (34-66), Weak (<34).">i</span>
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
                <span class="stock-detail-info stock-detail-tip" data-tooltip="Quality Score is a 0-100 weighted composite of percentile-ranked inputs: Revenue Growth (20%), EBITDA Margin (20%), ROE (20%), Free Cash Flow Margin (20%), Volatility Stability (10%), and Drawdown Stability (10%). Missing values are assigned a neutral percentile of 0.5. Tiers: Strong (>=67), Neutral (34-66), Weak (<34).">i</span>
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


def _render_performance_metrics_cards(summary: dict[str, object]) -> None:
    def _num(key: str) -> float | None:
        v = summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _fmt_pct(value: float | None, with_plus: bool = False) -> str:
        if value is None:
            return "N/A"
        out = value * 100.0
        if with_plus and out > 0:
            return f"+{out:.2f}%"
        return f"{out:.2f}%"

    def _fmt_num(value: float | None, decimals: int = 2) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{decimals}f}"

    def _badge(tone: str, text: str) -> str:
        cls = "pm-badge"
        if tone == "high":
            cls += " high"
        elif tone == "moderate":
            cls += " moderate"
        elif tone == "stat":
            cls += " stat"
        else:
            cls += " low"
        return f'<span class="{cls}">{html.escape(text)}</span>'

    vol = _num("volatility")
    if vol is None:
        vol_tone, vol_lbl = "low", "n/a"
    elif vol < 0.15:
        vol_tone, vol_lbl = "low", "low"
    elif vol < 0.30:
        vol_tone, vol_lbl = "moderate", "moderate"
    else:
        vol_tone, vol_lbl = "high", "high"

    corr = _num("correlation_with_benchmark")
    if corr is None:
        corr_tone, corr_lbl = "low", "n/a"
    elif corr >= 0.80:
        corr_tone, corr_lbl = "high", "high"
    elif corr >= 0.50:
        corr_tone, corr_lbl = "moderate", "moderate"
    else:
        corr_tone, corr_lbl = "low", "low"

    st.markdown("#### Performance Metrics")
    st.markdown(
        """
        <style>
        .pm-head { color: var(--muted); font-size: 0.86rem; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 10px; }
        .pm-top-title { color: var(--text); font-size: 1.15rem; font-weight: 700; margin-bottom: 2px; }
        .pm-top-value { color: var(--text); font-size: 2.55rem; font-weight: 900; line-height: 1.0; margin-bottom: 6px; }
        .pm-top-value.green { color: #79cf52; }
        .pm-top-note { color: var(--muted); font-size: 0.95rem; line-height: 1.3; }
        .pm-top-accent { height: 6px; border-radius: 999px; background: linear-gradient(90deg, #25c5ff, #2dd4a8); margin-top: 10px; }
        .pm-info {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text);
            background: var(--surface-2);
            border: 1px solid var(--border);
            cursor: help;
            user-select: none;
            position: relative;
            vertical-align: middle;
            margin-left: 6px;
        }
        .pm-info:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 0;
            bottom: calc(100% + 10px);
            width: min(460px, calc(100vw - 36px));
            background: #0a1324;
            color: #e8f1ff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 0.82rem;
            line-height: 1.35;
            font-weight: 500;
            text-align: left;
            white-space: normal;
            z-index: 9999;
            pointer-events: none;
            box-shadow: var(--shadow);
        }
        .pm-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; box-shadow: var(--shadow); margin-bottom: 14px; }
        .pm-subhead { color: var(--muted); font-size: 0.84rem; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; margin: 8px 0 2px 0; }
        .pm-row { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; padding: 10px 0; }
        .pm-row + .pm-row { border-top: 1px solid var(--border); }
        .pm-label { color: var(--text); font-size: 1.05rem; font-weight: 700; }
        .pm-desc { color: var(--muted); font-size: 0.90rem; line-height: 1.32; margin-top: 2px; max-width: 780px; }
        .pm-value { color: var(--text); font-size: 2.0rem; font-weight: 900; white-space: nowrap; margin-top: 2px; }
        .pm-value.negative { color: #ff8d8d; }
        .pm-value.subtle { font-size: 1.8rem; }
        .pm-badge { display: inline-block; margin-left: 8px; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 700; }
        .pm-badge.high { background: rgba(255, 172, 28, 0.18); color: #ffca6f; }
        .pm-badge.moderate { background: rgba(255, 186, 70, 0.18); color: #ffd989; }
        .pm-badge.low { background: rgba(64, 199, 129, 0.18); color: #8de3ba; }
        .pm-badge.stat { background: rgba(80, 145, 255, 0.20); color: #9dc1ff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pm-head">At A Glance</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            f'<div class="pm-top-title">Annual return (CAGR)</div><div class="pm-top-value green">{_fmt_pct(_num("CAGR"), with_plus=True)}</div><div class="pm-top-note">Average yearly growth rate</div>',
            unsafe_allow_html=True,
        )
        with g2:
            st.markdown(
                f'<div class="pm-top-title">Risk-adjusted return</div><div class="pm-top-value">{_fmt_num(_num("Sharpe_ratio"), 2)} <span style="font-size:0.55em; font-weight:700;">Sharpe</span><span class="pm-info" data-tooltip="Sharpe ratio tells you return per unit of risk. Formula used in this app: Sharpe = (CAGR - risk_free_rate) / volatility. Here, CAGR is annualized return, risk_free_rate is the simulator baseline return (auto-filled from treasury with 4% fallback unless you change it), and volatility is annualized return variability. Higher Sharpe generally means better risk-adjusted performance.">i</span></div><div class="pm-top-note">Above 1.0 is generally considered good</div><div class="pm-top-accent"></div>',
                unsafe_allow_html=True,
            )

    left_metrics_col, right_metrics_col = st.columns(2)
    with left_metrics_col:
        st.markdown('<div class="pm-head" style="margin-top:14px;">Risk - How Much Could You Lose?</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="pm-panel pm-panel-equal">
                <div class="pm-row">
                    <div>
                        <div class="pm-label">Volatility {_badge(vol_tone, vol_lbl)}</div>
                        <div class="pm-desc">How much the portfolio swings up and down on a typical year. Higher means a bumpier ride.</div>
                    </div>
                    <div class="pm-value subtle">{_fmt_pct(vol)}</div>
                </div>
                <div class="pm-row">
                    <div>
                        <div class="pm-label">Biggest drop ever (max drawdown)</div>
                        <div class="pm-desc">The largest peak-to-trough fall recorded. At worst, this is how far the portfolio was down.</div>
                    </div>
                    <div class="pm-value negative">{_fmt_pct(_num("max_drawdown"))}</div>
                </div>
                <div class="pm-row">
                    <div>
                        <div class="pm-label">Worst single day</div>
                        <div class="pm-desc">The steepest one-day loss in the dataset.</div>
                    </div>
                    <div class="pm-value negative">{_fmt_pct(_num("worst_day"))}</div>
                </div>
                <div class="pm-row">
                    <div>
                        <div class="pm-label">Worst single month</div>
                        <div class="pm-desc">The steepest one-month loss in the dataset.</div>
                    </div>
                    <div class="pm-value negative">{_fmt_pct(_num("worst_month"))}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_metrics_col:
        st.markdown('<div class="pm-head" style="margin-top:14px;">Tail Risk - Bad Day Estimates</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="pm-panel">
                <div class="pm-row">
                    <div>
                        <div class="pm-label">VaR 95% {_badge("stat", "statistical")}</div>
                        <div class="pm-desc">On a typical bad day (worst 5% of days), expected loss is at least this much.</div>
                    </div>
                    <div class="pm-value negative">{_fmt_pct(_num("VaR_95"))}</div>
                </div>
                <div class="pm-row">
                    <div>
                        <div class="pm-label">CVaR 95%</div>
                        <div class="pm-desc">On the very worst days (beyond VaR), this is the average loss.</div>
                    </div>
                    <div class="pm-value negative">{_fmt_pct(_num("CVaR_95"))}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="pm-head">Benchmark Comparison - How Does It Move Vs The Market?</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="pm-panel">
                <div class="pm-row">
                    <div>
                        <div class="pm-label">Correlation {_badge(corr_tone, corr_lbl)}</div>
                        <div class="pm-desc">How closely returns track the benchmark. 1.0 means it moves in lockstep with the market.</div>
                    </div>
                    <div class="pm-value">{_fmt_num(corr, 2)}</div>
                </div>
                <div class="pm-row">
                    <div>
                        <div class="pm-label">Beta</div>
                        <div class="pm-desc">Sensitivity to market moves. Higher beta means larger upside and downside swings.</div>
                    </div>
                    <div class="pm-value">{_fmt_num(_num("beta_relative_to_benchmark"), 2)} <span style="font-size:0.55em; font-weight:700;">vs market</span></div>
                </div>
            </div>
            """,
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


def _show_market_intelligence_tab() -> None:
    st.markdown("## Decision Intelligence")
    import json

    regime_df, _re = read_parquet_safe(REGIME_CACHE_PATH)
    risk_df, _rk = read_parquet_safe(RISK_CACHE_PATH)
    prob_df, _rp = read_parquet_safe(REGIME_PROB_PATH)
    evidence_df, _ee = read_parquet_safe(REGIME_EVIDENCE_PATH)
    alert_df, _ae = read_parquet_safe(ALERT_LOG_PATH)
    quality_df, _qe = read_parquet_safe(QUALITY_CACHE_PATH)

    def _latest_row(df: pd.DataFrame | None) -> pd.Series | None:
        if df is None or df.empty:
            return None
        out = df.copy()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out = out.dropna(subset=["Date"]).sort_values("Date")
            if out.empty:
                return None
        return out.tail(1).iloc[0]

    def _row_get(row: pd.Series | None, key: str, default: object) -> object:
        if row is None:
            return default
        return row.get(key, default)

    def _conf_bucket(v: float) -> tuple[str, str]:
        if not np.isfinite(v):
            return "Unknown", "#dce3ee"
        if v > 0.70:
            return "High", "#dff4ec"
        if v >= 0.40:
            return "Moderate", "#f3ecd9"
        return "Low", "#f3e2dd"

    rt, lt, pt, ps, po = st.tabs(["Is now a good time to invest?", "What looks promising?", "Portfolio suggestion", "Portfolio simulation", "Portfolio optimization"])

    with rt:
        regime_last = _latest_row(regime_df)
        risk_last = _latest_row(risk_df)
        prob_last = _latest_row(prob_df)
        ev_last = _latest_row(evidence_df)

        regime_label = str(_row_get(regime_last, "RegimeLabel", "Neutral"))
        mood_map = {
            "Risk On": ("GREEN", "Conditions look favorable", "Markets are in a risk-on environment", "#e1f0d6", "#2f5f2a"),
            "Neutral": ("YELLOW", "Conditions are mixed", "No clear bullish or bearish trend", "#f6efd7", "#6a5315"),
            "Risk Off": ("RED", "Conditions are defensive", "Markets are in a risk-off environment", "#f3dfde", "#7b2c2c"),
        }
        mood_tag, mood_title, mood_sub, mood_bg, mood_fg = mood_map.get(
            regime_label, ("YELLOW", "Conditions are mixed", "No clear bullish or bearish trend", "#f6efd7", "#6a5315")
        )

        risk_level = str(_row_get(risk_last, "RiskLevel", "Unknown"))

        conf_val = float("nan")
        if prob_last is not None and "ConfidenceScore" in prob_last.index:
            conf_val = float(pd.to_numeric(prob_last.get("ConfidenceScore"), errors="coerce"))
        if not np.isfinite(conf_val) and regime_last is not None and "ConfidenceScore" in regime_last.index:
            conf_val = float(pd.to_numeric(regime_last.get("ConfidenceScore"), errors="coerce"))
        conf_label, conf_bg = _conf_bucket(conf_val)

        summary_plain_map = {
            "Risk On": "Market tone is constructive. You can lean in selectively, but add risk in steps.",
            "Neutral": "Market tone is mixed. Hold steady, stay selective, and avoid big one-shot moves.",
            "Risk Off": "Market tone is defensive. Protect downside and keep new risk tighter.",
        }
        summary_line = summary_plain_map.get(regime_label, "The market is in a wait-and-see phase.")
        if ev_last is not None:
            short = str(ev_last.get("ShortExplanation", "") or "").strip()
            trigger_raw = str(ev_last.get("RuleTriggered", "") or "").strip()
            trigger_key = trigger_raw.lower()
            trigger_map = {
                "neutral_default": "No clear market trend is in place, so this is a hold-steady environment.",
                "risk_on_confirmed": "Risk appetite is improving; adding exposure gradually can be reasonable.",
                "risk_off_confirmed": "Downside pressure is elevated; favor defense and tighter risk controls.",
                "volatility_rising": "Volatility is rising; use smaller position sizes and stagger entries.",
                "yield_curve_inversion": "Macro stress signals are elevated; prioritize balance-sheet quality.",
            }
            jargon_tokens = [
                "risk-on",
                "risk off",
                "jointly satisfied",
                "trigger",
                "default",
                "conditions",
            ]
            short_lower = short.lower()
            short_is_geeky = any(tok in short_lower for tok in jargon_tokens)
            if short and not short_is_geeky:
                summary_line = short
            if trigger_key and trigger_key not in {"nan", "none"}:
                plain_trigger = trigger_map.get(trigger_key, "")
                if plain_trigger:
                    existing = summary_line.lower()
                    trigger_l = plain_trigger.lower()
                    redundant_pairs = [
                        ("hold steady", "hold-steady"),
                        ("no clear market trend", "mixed"),
                        ("defensive", "downside"),
                        ("add risk in steps", "adding exposure gradually"),
                    ]
                    is_redundant = any(a in existing and b in trigger_l for a, b in redundant_pairs) or any(
                        b in existing and a in trigger_l for a, b in redundant_pairs
                    )
                    if plain_trigger.rstrip(".").lower() in existing:
                        is_redundant = True
                    if not is_redundant:
                        summary_line = f"{summary_line} {plain_trigger}"

        fresh_count = 0
        stale_count = 0
        unknown_count = 0
        generated_at = pd.NaT
        try:
            with open(MODEL_HEALTH_PATH, "r", encoding="utf-8") as f:
                health = json.load(f)
            generated_at = pd.to_datetime(health.get("generated_at"), errors="coerce")
            freshness = health.get("model_freshness", {})
            if isinstance(freshness, dict):
                vals = [str(v).strip().lower() for v in freshness.values()]
                fresh_count = int(sum(v == "fresh" for v in vals))
                stale_count = int(sum(v == "stale" for v in vals))
                unknown_count = int(len(vals) - fresh_count - stale_count)
        except Exception:
            fresh_count = 0
            stale_count = 0
            unknown_count = 0

        # Cross-check with live cache status to avoid optimistic stale/refresh labels.
        live_paths = [
            REGIME_CACHE_PATH,
            RISK_CACHE_PATH,
            REGIME_PROB_PATH,
            REGIME_EVIDENCE_PATH,
            ALERT_LOG_PATH,
        ]
        live_statuses = [get_cache_status(p, MAX_AGE_DAYS) for p in live_paths]
        live_ready = int(sum(1 for s in live_statuses if s.exists and s.schema_ok and s.is_fresh))
        live_problem = int(len(live_statuses) - live_ready)

        latest_signal_date = pd.to_datetime(
            max(
                [
                    pd.to_datetime(_row_get(regime_last, "Date", pd.NaT), errors="coerce"),
                    pd.to_datetime(_row_get(risk_last, "Date", pd.NaT), errors="coerce"),
                    pd.to_datetime(_row_get(prob_last, "Date", pd.NaT), errors="coerce"),
                    pd.to_datetime(_row_get(ev_last, "Date", pd.NaT), errors="coerce"),
                ]
            ),
            errors="coerce",
        )
        refresh_date = pd.to_datetime(generated_at, errors="coerce")
        if pd.isna(refresh_date):
            refresh_date = latest_signal_date
        refresh_text = refresh_date.strftime("%b %d, %Y") if pd.notna(refresh_date) else "unknown date"
        today_local = pd.Timestamp.now().normalize()
        is_today_refresh = pd.notna(refresh_date) and (refresh_date.normalize() == today_local)

        if live_problem == 0 and fresh_count > 0 and stale_count == 0 and unknown_count == 0:
            data_status_text = (
                "Data is up to date - last refreshed today"
                if is_today_refresh
                else f"Data is up to date - last refreshed {refresh_text}"
            )
            data_bg = "#e1f0d6"
            data_fg = "#2f5f2a"
        else:
            data_status_text = (
                f"Data needs refresh - {live_problem} of {len(live_statuses)} live caches are stale/missing"
                if live_problem > 0
                else "Data needs refresh - some model artifacts are stale"
            )
            data_bg = "#f6efd7"
            data_fg = "#6a5315"

        critical_count = 0
        if alert_df is not None and not alert_df.empty and "Severity" in alert_df.columns:
            adf = alert_df.copy()
            if "Date" in adf.columns:
                adf["Date"] = pd.to_datetime(adf["Date"], errors="coerce")
                adf = adf.dropna(subset=["Date"])
                if not adf.empty:
                    ref_date = pd.to_datetime(max(adf["Date"].max(), latest_signal_date), errors="coerce")
                    if pd.notna(ref_date):
                        adf = adf[adf["Date"] >= (ref_date - pd.Timedelta(days=90))]
            critical_count = int((adf["Severity"].astype(str).str.lower() == "critical").sum())

        conf_pct_val = int(round(conf_val * 100)) if np.isfinite(conf_val) else 0
        conf_pct = f"{conf_pct_val}%"
        conf_bar = max(6, min(100, conf_pct_val))
        risk_badge = str(risk_level).strip().lower()
        risk_pill = {"low": ("Safe", "#daf5ea", "#1f6a4c"), "moderate": ("Watch", "#f8efd9", "#80550e"), "high": ("Elevated", "#f5dfdf", "#7b2c2c")}
        risk_txt, risk_bg, risk_fg = risk_pill.get(risk_badge, ("Unknown", "#dce3ee", "#2f3e58"))
        mood_title_emph = {"Risk On": "Favorable", "Neutral": "Mixed", "Risk Off": "Defensive"}.get(regime_label, "Mixed")
        mood_icon_map = {
            "Risk On": ("R+", "#25d3a8", "#0b2a2b"),
            "Neutral": ("N", "#ffd247", "#2a2208"),
            "Risk Off": ("R-", "#ff8d8d", "#2d1212"),
        }
        mood_icon_text, mood_icon_bg, mood_icon_fg = mood_icon_map.get(regime_label, ("N", "#ffd247", "#2a2208"))

        h_left, h_right = st.columns([1.2, 1.0], vertical_alignment="bottom")
        with h_left:
            st.markdown(
                f"""
                <div style="display:inline-block; border:1px solid #6d5d20; border-radius:10px; padding:5px 10px; background:rgba(38,27,7,0.55); color:#f3c445; font-weight:800; letter-spacing:0.05em; font-size:0.86rem; margin-bottom:6px;">
                  ⚡ {html.escape(regime_label.upper())} SIGNAL
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h_right:
            st.markdown(
                "<div style='font-size:2.0rem; font-weight:800; color:#edf4ff; margin:0 0 6px 0;'>Key Metrics</div>",
                unsafe_allow_html=True,
            )

        left_col, right_col = st.columns([1.2, 1.0], vertical_alignment="top")
        with left_col:
            st.markdown(
                f"""
                <div style="border:1px solid rgba(120,146,181,0.24); border-radius:16px; padding:14px 16px; background:linear-gradient(140deg, rgba(9,20,39,0.92), rgba(9,20,39,0.65)); min-height:154px;">
                  <div style="display:flex; gap:14px; align-items:center;">
                    <div style="width:72px; height:72px; border-radius:16px; background:linear-gradient(145deg, rgba(20,34,56,0.95), rgba(11,22,40,0.95)); border:1px solid rgba(129,156,193,0.35); display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px;">
                      <div style="padding:3px 9px; border-radius:999px; background:{mood_icon_bg}; color:{mood_icon_fg}; font-size:0.86rem; font-weight:900; letter-spacing:0.03em;">{mood_icon_text}</div>
                      <div style="width:28px; height:3px; border-radius:999px; background:rgba(120,146,181,0.55);"></div>
                    </div>
                    <div>
                      <div style="font-size:2.0rem; font-weight:900; line-height:1.04; color:#f3f7ff;">Conditions are <span style="color:#ffd247;">{html.escape(mood_title_emph)}</span></div>
                    </div>
                  </div>
                  <div style="margin-top:12px; color:#b5c8e6; font-size:0.95rem;">{html.escape(mood_sub)}.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right_col:
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(120,146,181,0.24); border-top:3px solid #37d6b5; border-radius:14px; padding:10px 12px; background:rgba(10,22,41,0.50); min-height:154px;">
                      <div class="diw-muted">Risk level</div>
                      <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                        <div style="font-size:1.75rem; font-weight:900; color:#edf4ff;">{html.escape(risk_level)}</div>
                        <div style="padding:2px 8px; border-radius:999px; background:{risk_bg}; color:{risk_fg}; font-weight:700; font-size:0.82rem;">{risk_txt}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(120,146,181,0.24); border-top:3px solid #a26dff; border-radius:14px; padding:10px 12px; background:rgba(10,22,41,0.50); min-height:154px;">
                      <div class="diw-muted">Confidence</div>
                      <div style="display:flex; align-items:center; justify-content:space-between; margin-top:4px;">
                        <div style="font-size:1.75rem; font-weight:900; color:#edf4ff;">{conf_pct}</div>
                        <div style="padding:2px 8px; border-radius:999px; background:{conf_bg}; color:#4f401f; font-weight:700; font-size:0.82rem;">{conf_label}</div>
                      </div>
                      <div style="height:5px; margin-top:12px; border-radius:999px; background:rgba(131,152,182,0.26);">
                        <div style="height:5px; width:{conf_bar}%; border-radius:999px; background:linear-gradient(90deg, #26d5c3, #6f7cff);"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"""
            <div style="margin-top:10px; border:1px solid rgba(120,146,181,0.24); border-top:3px solid #26d5c3; border-radius:14px; padding:14px 16px; background:rgba(10,22,41,0.45);">
              <div style="color:#26d5c3; font-size:1.05rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em;">What This Means For You</div>
              <div style="margin-top:8px; color:#d6e3f8; font-size:1.12rem; line-height:1.55;">{html.escape(summary_line)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Data Status")
        st.markdown(
            f"""
            <div style="border:1px solid rgba(60,190,150,0.35); border-radius:12px; padding:12px 14px; background:linear-gradient(90deg, rgba(22,72,52,0.55), rgba(8,33,49,0.35)); color:#8ef3be; font-weight:800;">
              ✔ {html.escape(data_status_text)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if critical_count > 0:
            st.markdown(
                f"""
                <div style="margin-top:9px; border:1px solid rgba(250,184,65,0.45); border-radius:12px; padding:12px 14px; background:linear-gradient(90deg, rgba(72,48,16,0.60), rgba(34,24,7,0.35)); color:#ffcf67; font-weight:800;">
                  ⚠ {critical_count} critical signal{'s' if critical_count != 1 else ''} detected - review before acting
                </div>
                """,
                unsafe_allow_html=True,
            )

    with lt:
        st.markdown("### Top Opportunities")
        if quality_df is None or quality_df.empty:
            st.caption("Quality data is not available right now.")
        else:
            q = quality_df.copy()
            q["QualityScore"] = pd.to_numeric(q.get("QualityScore"), errors="coerce")
            q["Ticker"] = q.get("Ticker", "").astype(str).str.upper().str.strip()
            q["QualityTier"] = q.get("QualityTier", "").astype(str)
            q = q.dropna(subset=["QualityScore"])
            q = q[q["Ticker"] != ""]
            ranked = q.sort_values("QualityScore", ascending=False).head(30).copy()
            if ranked.empty:
                st.caption("No candidates available in cache.")
            else:
                fdf = _load_fundamentals_union()
                company_map: dict[str, str] = {}
                if fdf is not None and not fdf.empty and {"Ticker", "Company"}.issubset(set(fdf.columns)):
                    ftmp = fdf[["Ticker", "Company"]].copy()
                    ftmp["Ticker"] = ftmp["Ticker"].astype(str).str.upper().str.strip()
                    ftmp["Company"] = ftmp["Company"].astype(str).str.strip()
                    company_map = dict(zip(ftmp["Ticker"], ftmp["Company"]))

                qu, _qerr = read_parquet_safe(QUALITY_UNCERTAINTY_PATH)
                conf_map: dict[str, str] = {}
                if qu is not None and not qu.empty and "Ticker" in qu.columns:
                    uq = qu.copy()
                    uq["Ticker"] = uq["Ticker"].astype(str).str.upper().str.strip()
                    if "Date" in uq.columns:
                        uq["Date"] = pd.to_datetime(uq["Date"], errors="coerce")
                        uq = uq.sort_values("Date")
                    for tkr, grp in uq.groupby("Ticker"):
                        r = grp.tail(1).iloc[0]
                        stability = float(pd.to_numeric(r.get("TierStability"), errors="coerce")) if "TierStability" in r else float("nan")
                        p10 = float(pd.to_numeric(r.get("ScoreP10"), errors="coerce")) if "ScoreP10" in r else float("nan")
                        p90 = float(pd.to_numeric(r.get("ScoreP90"), errors="coerce")) if "ScoreP90" in r else float("nan")
                        if np.isfinite(stability):
                            cscore = stability
                        elif np.isfinite(p10) and np.isfinite(p90):
                            cscore = max(0.0, 1.0 - min(1.0, (p90 - p10) / 10.0))
                        else:
                            cscore = float("nan")
                        if np.isfinite(cscore) and cscore > 0.80:
                            conf_map[str(tkr)] = "High"
                        elif np.isfinite(cscore) and cscore >= 0.60:
                            conf_map[str(tkr)] = "Med"
                        else:
                            conf_map[str(tkr)] = "Low"

                def _stars(score: float) -> int:
                    if score >= 75:
                        return 5
                    if score >= 68:
                        return 4
                    if score >= 60:
                        return 3
                    if score >= 50:
                        return 2
                    return 1

                rows: list[dict[str, object]] = []
                for idx, (_, row) in enumerate(ranked.iterrows(), start=1):
                    tkr = str(row.get("Ticker", ""))
                    score = float(pd.to_numeric(row.get("QualityScore"), errors="coerce"))
                    company = company_map.get(tkr, "Candidate")
                    tier = str(row.get("QualityTier", "Neutral")).strip().lower()
                    stars = ("★" * _stars(score)) + ("☆" * (5 - _stars(score)))
                    if tier == "strong":
                        tier_lbl = "Top tier"
                    elif tier == "neutral":
                        tier_lbl = "Mid tier"
                    else:
                        tier_lbl = "Watch"
                    rows.append(
                        {
                            "Rank": idx,
                            "Ticker": tkr,
                            "Company": company,
                            "Stars": stars,
                            "Tier": tier_lbl,
                            "Confidence": conf_map.get(tkr, "Med"),
                            "QualityScore": round(score, 1),
                        }
                    )

                show = pd.DataFrame(rows, columns=["Rank", "Ticker", "Company", "Stars", "Tier", "Confidence", "QualityScore"])
                selected_ticker = None
                if AgGrid is not None and GridOptionsBuilder is not None and JsCode is not None:
                    show_grid = show.copy()
                    show_grid.insert(0, "Action", "View Details")
                    gb = GridOptionsBuilder.from_dataframe(show_grid)
                    gb.configure_default_column(sortable=True, resizable=True, flex=1, minWidth=120)
                    for c in ["Action", "Rank", "Ticker", "Company", "Stars", "Tier", "Confidence", "QualityScore"]:
                        gb.configure_column(
                            c,
                            cellStyle={"textAlign": "center"},
                            headerClass="di-header-center",
                            cellClass="di-cell-center",
                        )
                    button_renderer = JsCode(
                        """
                        class BtnCellRenderer {
                          init(params) {
                            this.eGui = document.createElement('button');
                            this.eGui.innerText = 'View Details';
                            this.eGui.style.padding = '2px 10px';
                            this.eGui.style.borderRadius = '999px';
                            this.eGui.style.border = '1px solid rgba(91,186,255,0.45)';
                            this.eGui.style.background = 'rgba(34,80,130,0.28)';
                            this.eGui.style.color = '#dff1ff';
                            this.eGui.style.fontWeight = '700';
                            this.eGui.style.cursor = 'pointer';
                            this.eGui.addEventListener('click', () => {
                              params.node.setSelected(true, true);
                            });
                          }
                          getGui() { return this.eGui; }
                        }
                        """
                    )
                    gb.configure_column(
                        "Action",
                        headerName="Details",
                        width=150,
                        minWidth=130,
                        maxWidth=170,
                        sortable=False,
                        filter=False,
                        cellRenderer=button_renderer,
                    )
                    gb.configure_selection("single", use_checkbox=False)
                    grid_options = gb.build()
                    grid_options["pagination"] = True
                    grid_options["paginationPageSize"] = 5
                    grid_options["rowHeight"] = 36
                    grid_options["headerHeight"] = 40
                    grid_height = int(40 + (5 * 36) + 44 + 2)
                    aggrid_kwargs = {}
                    if GridUpdateMode is not None:
                        aggrid_kwargs["update_mode"] = GridUpdateMode.SELECTION_CHANGED
                    grid_resp = AgGrid(
                        show_grid,
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
                            },
                        },
                        **aggrid_kwargs,
                    )
                    sel = grid_resp.get("selected_rows", [])
                    if isinstance(sel, pd.DataFrame) and not sel.empty:
                        selected_ticker = str(sel.iloc[0].get("Ticker", "")).strip().upper()
                    elif isinstance(sel, list) and sel:
                        selected_ticker = str((sel[0] or {}).get("Ticker", "")).strip().upper()
                    elif isinstance(sel, dict):
                        selected_ticker = str(sel.get("Ticker", "")).strip().upper()
                else:
                    _render_sortable_centered_table(show, ["Rank", "Stars", "Tier", "Confidence", "QualityScore"], page_size=5)

                st.markdown("#### Selected Pick Details")
                if selected_ticker:
                    pick = selected_ticker
                else:
                    pick = st.selectbox(
                        "Select from displayed picks",
                        options=show["Ticker"].astype(str).tolist(),
                        index=0,
                        key="mi_pick_detail_ticker",
                    )
                if pick:
                    qrow = ranked[ranked["Ticker"].astype(str) == str(pick)].head(1)
                    if not qrow.empty:
                        qrow = qrow.iloc[0]
                        company = company_map.get(str(pick), "Company")
                        qscore = float(pd.to_numeric(qrow.get("QualityScore"), errors="coerce"))
                        qtier = str(qrow.get("QualityTier", "Neutral"))

                        feat_map = {
                            "Revenue_Growth_YoY_Pct": "Stable revenue growth",
                            "EBITDA_Margin": "Strong earnings consistency",
                            "ROE": "Efficient use of shareholder capital",
                            "FreeCashFlow_Margin": "Healthy cash flow quality",
                            "Volatility_63D_stability": "Stable price behavior",
                            "Drawdown_252D_stability": "Lower downside stress",
                        }

                        pos_list: list[str] = []
                        neg_list: list[str] = []
                        qx, _qxe = read_parquet_safe(QUALITY_EXPLAIN_PATH)
                        if qx is not None and not qx.empty and {"Ticker", "ContributionJSON"}.issubset(set(qx.columns)):
                            xx = qx.copy()
                            xx["Ticker"] = xx["Ticker"].astype(str).str.upper().str.strip()
                            rr = xx[xx["Ticker"] == str(pick)].tail(1)
                            if not rr.empty:
                                try:
                                    contrib = json.loads(str(rr.iloc[0].get("ContributionJSON", "{}")))
                                except Exception:
                                    contrib = {}
                                items: list[tuple[str, float]] = []
                                for k, v in (contrib or {}).items():
                                    vv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
                                    if pd.isna(vv):
                                        continue
                                    items.append((str(k), float(vv)))
                                pos_items = sorted([x for x in items if x[1] > 0], key=lambda x: abs(x[1]), reverse=True)[:3]
                                neg_items = sorted([x for x in items if x[1] < 0], key=lambda x: abs(x[1]), reverse=True)[:3]
                                pos_list = [feat_map.get(k, k.replace("_", " ")) for k, _ in pos_items]
                                neg_list = [feat_map.get(k, k.replace("_", " ")) for k, _ in neg_items]

                        p10 = p50 = p90 = float("nan")
                        tier_stable = ""
                        qu2, _qu2e = read_parquet_safe(QUALITY_UNCERTAINTY_PATH)
                        if qu2 is not None and not qu2.empty and "Ticker" in qu2.columns:
                            uu = qu2.copy()
                            uu["Ticker"] = uu["Ticker"].astype(str).str.upper().str.strip()
                            if "Date" in uu.columns:
                                uu["Date"] = pd.to_datetime(uu["Date"], errors="coerce")
                                uu = uu.sort_values("Date")
                            ur = uu[uu["Ticker"] == str(pick)].tail(1)
                            if not ur.empty:
                                ur = ur.iloc[0]
                                p10 = float(pd.to_numeric(ur.get("ScoreP10"), errors="coerce"))
                                p50 = float(pd.to_numeric(ur.get("ScoreP50"), errors="coerce"))
                                p90 = float(pd.to_numeric(ur.get("ScoreP90"), errors="coerce"))
                                ts = float(pd.to_numeric(ur.get("TierStability"), errors="coerce")) if "TierStability" in ur else float("nan")
                                if np.isfinite(ts):
                                    tier_stable = "Yes" if ts >= 0.67 else "No"

                        if not np.isfinite(p50):
                            p50 = qscore
                        if not np.isfinite(p10):
                            p10 = max(0.0, p50 - 10.0)
                        if not np.isfinite(p90):
                            p90 = min(100.0, p50 + 10.0)
                        band = max(0.0, p90 - p10)
                        conf_txt = "High confidence" if band < 15 else ("Medium confidence" if band <= 30 else "Low confidence")
                        p10w = max(4.0, min(100.0, p10))
                        p50w = max(4.0, min(100.0, p50))
                        p90w = max(4.0, min(100.0, p90))
                        pos_lines = "".join([f"<div style='color:#59b133; margin:4px 0;'>+ {html.escape(x)}</div>" for x in (pos_list or ["Strengths are broad-based across fundamentals"])])
                        neg_lines = "".join([f"<div style='color:#d14a4a; margin:4px 0;'>- {html.escape(x)}</div>" for x in (neg_list or ["No major negative driver stands out"])])

                        st.markdown(
                            f"""
                            <div style="border:1px solid rgba(120,146,181,0.24); border-radius:14px; padding:14px 16px; background:linear-gradient(180deg, rgba(11,22,40,0.62), rgba(11,22,40,0.50));">
                              <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
                                <div>
                                  <div style="font-size:2.0rem; font-weight:900; color:#edf4ff;">{html.escape(str(pick))} - {html.escape(company)}</div>
                                  <div style="color:#b9cee8; margin-top:2px;">Quality score: {qscore:.1f} | {html.escape(qtier)}</div>
                                </div>
                                <div style="padding:4px 10px; border-radius:999px; background:#dff4ec; color:#1f6a4c; font-weight:800;">{html.escape(str(qtier).title())} tier</div>
                              </div>
                              <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:12px; border-top:1px solid rgba(120,146,181,0.18); padding-top:10px;">
                                <div><div style="color:#9db3d6; font-weight:800; text-transform:uppercase;">Working in its favour</div>{pos_lines}</div>
                                <div><div style="color:#9db3d6; font-weight:800; text-transform:uppercase;">Working against it</div>{neg_lines}</div>
                              </div>
                              <div style="margin-top:10px; border-top:1px solid rgba(120,146,181,0.18); padding-top:10px;">
                                <div style="color:#9db3d6; font-weight:800; text-transform:uppercase;">How certain is this rating?</div>
                                <div style="margin-top:6px; display:grid; grid-template-columns: 160px 1fr 40px; gap:8px; align-items:center;">
                                  <div>Conservative case</div><div style="height:10px; border-radius:999px; background:rgba(131,152,182,0.20);"><div style="height:10px; width:{p10w:.1f}%; border-radius:999px; background:#b5d98f;"></div></div><div style="text-align:right;">{p10:.0f}</div>
                                  <div>Central estimate</div><div style="height:10px; border-radius:999px; background:rgba(131,152,182,0.20);"><div style="height:10px; width:{p50w:.1f}%; border-radius:999px; background:#79b83a;"></div></div><div style="text-align:right;">{p50:.0f}</div>
                                  <div>Optimistic case</div><div style="height:10px; border-radius:999px; background:rgba(131,152,182,0.20);"><div style="height:10px; width:{p90w:.1f}%; border-radius:999px; background:#5ea01f;"></div></div><div style="text-align:right;">{p90:.0f}</div>
                                </div>
                                <div style="margin-top:8px; color:#b9cee8;">Band width: {band:.0f} pts - {conf_txt}{(' - Tier stable: ' + tier_stable) if tier_stable else ''}</div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with pt:
        st.markdown("### Portfolio Suggestion")
        regime_last = _latest_row(regime_df)
        risk_last = _latest_row(risk_df)
        regime_label = str(_row_get(regime_last, "RegimeLabel", "Neutral"))
        risk_level = str(_row_get(risk_last, "RiskLevel", "Unknown"))

        if quality_df is None or quality_df.empty:
            st.caption("Portfolio suggestion unavailable because quality data is missing.")
        else:
            merged = _build_portfolio_suggestion_base()
            if merged is None or merged.empty:
                st.caption("Portfolio suggestion unavailable because source data is empty.")
                return

            st.markdown("#### Build Your Suggested Portfolio")
            c1, c2, c3 = st.columns(3)
            with c1:
                invest_type = st.selectbox(
                    "Investment type",
                    options=["Growth", "Income", "Defensive"],
                    index=0,
                    key="mi_invest_type",
                )
            with c2:
                sectors = sorted([s for s in merged.get("Sector", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()])
                selected_sector = st.selectbox(
                    "Sector filter",
                    options=["All sectors"] + sectors,
                    index=0,
                    key="mi_sector_filter",
                )
            with c3:
                weight_rule = st.selectbox(
                    "Weighting rule",
                    options=["Equal", "Score-proportional", "Manual"],
                    index=0,
                    key="mi_weight_rule",
                )

            # Step 1 + 2: investment-type and sector filters
            pool = merged.copy()
            notes: list[str] = []
            if invest_type == "Growth":
                pool = pool[(pool["QualityTier"].str.lower() == "strong") & (pool["QualityScore"] >= 67)]
            elif invest_type == "Income":
                if "DividendYield" in pool.columns:
                    pool["DividendYield"] = pd.to_numeric(pool["DividendYield"], errors="coerce")
                    pool = pool[pool["DividendYield"].fillna(0.0) > 0]
                else:
                    pool["EBITDA_Margin"] = pd.to_numeric(pool.get("EBITDA_Margin"), errors="coerce")
                    thresh = float(pool["EBITDA_Margin"].median()) if pool["EBITDA_Margin"].notna().any() else 0.0
                    pool = pool[pool["EBITDA_Margin"].fillna(-999) >= thresh]
                    notes.append("Income filter uses profitability proxy because DividendYield is unavailable in current cache.")
            else:  # Defensive
                pool["MarketCap"] = pd.to_numeric(pool.get("MarketCap"), errors="coerce")
                mc_thresh = float(pool["MarketCap"].median()) if pool["MarketCap"].notna().any() else float("nan")
                pool = pool[pool["QualityTier"].str.lower().isin(["strong", "neutral"])]
                if np.isfinite(mc_thresh):
                    pool = pool[pool["MarketCap"].fillna(0.0) >= mc_thresh]
                notes.append("Defensive filter uses higher-quality, larger-cap proxies (stock-level RiskScore is not available in cache).")

            if selected_sector != "All sectors" and "Sector" in pool.columns:
                pool = pool[pool["Sector"].astype(str) == selected_sector]

            # Step 3: top 10 by same ranking logic
            pool = pool.sort_values("QualityScore", ascending=False)
            picks = pool.head(10).copy()

            if picks.empty:
                st.warning("No stocks match the selected filters. Try broadening sector or investment type.")
            else:
                # Step 4: weighting rule
                if weight_rule == "Equal":
                    w = np.repeat(1.0 / len(picks), len(picks))
                elif weight_rule == "Score-proportional":
                    s = pd.to_numeric(picks["QualityScore"], errors="coerce").fillna(0.0)
                    s = np.maximum(s - s.min() + 1.0, 0.0)
                    w = (s / s.sum()).values if s.sum() > 0 else np.repeat(1.0 / len(picks), len(picks))
                else:
                    default_manual = ", ".join([str(round(100.0 / len(picks), 1))] * len(picks))
                    manual_txt = st.text_input(
                        "Manual weights (%) - comma-separated, in ticker order shown below",
                        value=default_manual,
                        key="mi_manual_weights",
                    )
                    toks = [t.strip() for t in str(manual_txt).split(",") if t.strip()]
                    vals: list[float] = []
                    ok = True
                    if len(toks) != len(picks):
                        ok = False
                    else:
                        for t in toks:
                            try:
                                vals.append(float(t))
                            except Exception:
                                ok = False
                                break
                    if ok and sum(vals) > 0:
                        arr = np.array(vals, dtype=float)
                        w = arr / arr.sum()
                    else:
                        st.caption("Invalid manual weights. Falling back to equal weights.")
                        w = np.repeat(1.0 / len(picks), len(picks))

                picks = picks.assign(WeightPct=np.round(w * 100.0, 1))
                drift = round(100.0 - float(picks["WeightPct"].sum()), 1)
                if abs(drift) >= 0.1:
                    picks.loc[picks.index[0], "WeightPct"] = round(float(picks.iloc[0]["WeightPct"]) + drift, 1)

                # Step 5: display with plain-English inclusion reason
                def _reason(row: pd.Series) -> str:
                    exp = str(row.get("Explanation", "") or "").strip()
                    first = exp.split(",")[0].strip() if exp else ""
                    if first:
                        return f"Included for {first.lower()}."
                    if invest_type == "Growth":
                        return "Included for strong quality and growth profile."
                    if invest_type == "Income":
                        return "Included for steadier profitability profile."
                    return "Included for resilient quality characteristics."

                picks["Reason"] = picks.apply(_reason, axis=1)
                show_cols = ["Ticker", "Sector", "WeightPct", "QualityTier", "Reason"]
                if "Sector" not in picks.columns:
                    picks["Sector"] = "N/A"
                show = picks[show_cols].rename(columns={"WeightPct": "Weight (%)", "QualityTier": "Tier"})

                for n in notes:
                    st.caption(n)

                _render_sortable_centered_table(show, ["Weight (%)", "Tier"], page_size=10)

                if st.button("Save Portfolio", key="mi_save_portfolio", use_container_width=False):
                    out_path = DATA_DIR / "portfolio_suggestions_saved.jsonl"
                    payload = {
                        "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
                        "investment_type": invest_type,
                        "sector_filter": selected_sector,
                        "weight_rule": weight_rule,
                        "market_mood": regime_label,
                        "risk_level": risk_level,
                        "holdings": show.to_dict(orient="records"),
                    }
                    try:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload) + "\n")
                        # Keep simulation holdings in sync with the just-saved portfolio.
                        synced_rows = _clean_sim_holdings_rows(
                            [
                                {
                                    "Ticker": str(r.get("Ticker", "")).upper().strip(),
                                    "WeightPct": float(pd.to_numeric(pd.Series([r.get("Weight (%)")]), errors="coerce").iloc[0] or 0.0),
                                }
                                for r in payload.get("holdings", [])
                            ]
                        )
                        if synced_rows:
                            st.session_state["mi_sim_holdings_rows"] = synced_rows
                            st.session_state["mi_sim_holdings_source"] = "Loaded from latest saved portfolio suggestion"
                            st.session_state["mi_sim_weighting_mode"] = "Manual weights"
                            # Also sync the main Portfolio Decision Simulator tab if user opens it next.
                            st.session_state["sim_holdings_rows"] = synced_rows
                            st.session_state["sim_weighting_mode"] = "Manual weights"
                        st.success(f"Saved to {out_path}")
                    except Exception as e:
                        st.error(f"Could not save portfolio: {e}")

    with ps:
        st.markdown("### Portfolio Simulation")

        def _load_latest_saved_holdings_rows() -> tuple[list[dict[str, object]], str]:
            out_path = DATA_DIR / "portfolio_suggestions_saved.jsonl"
            if not out_path.exists():
                return _default_sim_holdings_rows(), "Default starter holdings"
            try:
                lines = [ln.strip() for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    return _default_sim_holdings_rows(), "Default starter holdings"
                payload = json.loads(lines[-1])
                holdings = payload.get("holdings", [])
                rows: list[dict[str, object]] = []
                if isinstance(holdings, list):
                    for h in holdings:
                        t = str((h or {}).get("Ticker") or "").strip().upper()
                        w = pd.to_numeric(
                            pd.Series([(h or {}).get("Weight (%)", (h or {}).get("WeightPct"))]),
                            errors="coerce",
                        ).iloc[0]
                        if not t or pd.isna(w):
                            continue
                        rows.append({"Ticker": t, "WeightPct": float(w)})
                clean = _clean_sim_holdings_rows(rows)
                if clean:
                    return clean, "Loaded from latest saved portfolio suggestion"
            except Exception:
                pass
            return _default_sim_holdings_rows(), "Default starter holdings"

        price_status = get_cache_status(PRICES_CACHE_PATH, MAX_AGE_DAYS, required_columns=PRICE_SCHEMA_COLUMNS)
        if (not price_status.exists) or (not price_status.schema_ok):
            st.warning("Price cache not found or invalid. Run the scheduled pipeline before using the simulator.")
        else:
            fundamentals = _load_fundamentals_union()

            if "mi_sim_holdings_rows" not in st.session_state:
                init_rows, init_src = _load_latest_saved_holdings_rows()
                st.session_state["mi_sim_holdings_rows"] = init_rows
                st.session_state["mi_sim_holdings_source"] = init_src
                if str(init_src).lower().startswith("loaded from latest saved"):
                    st.session_state["mi_sim_weighting_mode"] = "Manual weights"
            st.session_state["mi_sim_holdings_rows"] = _clean_sim_holdings_rows(st.session_state.get("mi_sim_holdings_rows"))

            auto_rf, auto_src = _auto_risk_free_rate_pct()
            if "mi_sim_risk_free_rate_pct" not in st.session_state:
                st.session_state["mi_sim_risk_free_rate_pct"] = float(auto_rf)
                st.session_state["mi_sim_risk_free_rate_source"] = auto_src

            run_clicked = False
            export_clicked = False
            left_group, right_group = st.columns([2.4, 2.4], vertical_alignment="top")
            with right_group:
                act_a, act_b = st.columns(2, vertical_alignment="bottom")
                with act_a:
                    run_clicked = st.button("Run Simulation", key="mi_sim_run_btn", use_container_width=True)
                with act_b:
                    export_clicked = st.button("Export Decision Brief", key="mi_sim_export_brief", use_container_width=True)
                top_a, top_b = st.columns(2, vertical_alignment="bottom")
                with top_a:
                    weighting_mode = st.selectbox(
                        "Weighting",
                        ["Equal weight", "Market cap weight", "Manual weights"],
                        key="mi_sim_weighting_mode",
                    )
                with top_b:
                    benchmark = st.selectbox(
                        "Benchmark",
                        SIM_BENCHMARK_OPTIONS,
                        index=0,
                        key="mi_sim_benchmark",
                    )

                mid_a, mid_b = st.columns(2, vertical_alignment="bottom")
                with mid_a:
                    lookback_years = st.selectbox("Lookback period", [1, 3, 5, 10], index=2, key="mi_sim_lookback")
                with mid_b:
                    rebalance_label = st.selectbox("Rebalance", ["None", "Monthly"], index=0, key="mi_sim_rebalance")

                bot_a, bot_b, bot_c, bot_d = st.columns([1.05, 1.20, 1.10, 1.10], vertical_alignment="bottom")
                with bot_a:
                    mode_label = st.selectbox("Simulation mode", ["Historical", "Monte Carlo"], index=0, key="mi_sim_mode")
                with bot_b:
                    initial_capital = float(
                        st.number_input(
                            "Starting capital ($)",
                            min_value=100.0,
                            max_value=100000000.0,
                            value=10000.0,
                            step=1000.0,
                            key="mi_sim_initial_capital",
                        )
                    )
                with bot_c:
                    risk_free_rate_pct = float(
                        st.number_input(
                            "Risk-free rate (%)",
                            min_value=0.0,
                            max_value=15.0,
                            step=0.1,
                            key="mi_sim_risk_free_rate_pct",
                        )
                    )
                with bot_d:
                    strict_mode = st.checkbox("Strict missing-data mode", value=False, key="mi_sim_strict")
                src_note = str(st.session_state.get("mi_sim_risk_free_rate_source", auto_src))
                st.caption(f"Risk-free rate source: {risk_free_rate_pct:.2f}% ({src_note})")

            with left_group:
                st.caption("Portfolio holdings")
                popover_label = "Edit holdings list"
                popover_ctx = (
                    st.popover(popover_label, use_container_width=True)
                    if hasattr(st, "popover")
                    else st.expander(popover_label, expanded=False)
                )
                with popover_ctx:
                    editor_df = pd.DataFrame(st.session_state.get("mi_sim_holdings_rows", []), columns=["Ticker", "WeightPct"])
                    editor_kwargs: dict[str, object] = {}
                    if hasattr(st, "column_config"):
                        editor_kwargs["column_config"] = {
                            "Ticker": st.column_config.TextColumn("Ticker", help="Example: AAPL"),
                            "WeightPct": st.column_config.NumberColumn("Weight (%)", min_value=0.0, step=0.5, format="%.2f"),
                        }
                    edited_df = st.data_editor(
                        editor_df,
                        use_container_width=True,
                        hide_index=True,
                        num_rows="dynamic",
                        key="mi_sim_holdings_editor",
                        **editor_kwargs,
                    )
                    proposed_rows = _clean_sim_holdings_rows(edited_df.to_dict(orient="records"))
                    if st.button("Apply holdings", key="mi_sim_apply_holdings", use_container_width=True):
                        st.session_state["mi_sim_holdings_rows"] = proposed_rows
                        st.session_state["mi_sim_weighting_mode"] = "Manual weights"
                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:
                            st.experimental_rerun()

                sim_rows = st.session_state.get("mi_sim_holdings_rows", [])
                ticker_text, manual_weights_text, total_weight_pct = _sim_rows_to_inputs(sim_rows)
                st.caption(f"Holdings: {len(sim_rows)} | Total entered: {total_weight_pct:.2f}% | {str(st.session_state.get('mi_sim_holdings_source', ''))}")
                if sim_rows:
                    tickers_preview = _parse_ticker_input(ticker_text)
                    effective_preview, preview_warnings = _build_holdings(
                        tickers=tickers_preview,
                        weighting_mode=weighting_mode,
                        fundamentals_df=fundamentals,
                        manual_weights_text=manual_weights_text,
                    )
                    if effective_preview:
                        preview_df = pd.DataFrame(
                            [{"Ticker": t, "Weight (%)": round(float(w) * 100.0, 2)} for t, w in effective_preview]
                        )
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                        st.caption(f"Displayed allocation reflects: {weighting_mode}")
                    else:
                        preview_df = pd.DataFrame(sim_rows, columns=["Ticker", "WeightPct"]).rename(columns={"WeightPct": "Weight (%)"})
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    if preview_warnings:
                        st.caption(preview_warnings[0])

            mc_paths = 1000
            horizon_days = 252
            if mode_label == "Monte Carlo":
                m1, m2 = st.columns(2, vertical_alignment="bottom")
                with m1:
                    mc_paths = int(st.number_input("Number of simulations", min_value=100, max_value=20000, value=1000, step=100, key="mi_sim_mc_paths"))
                with m2:
                    horizon_years = float(st.number_input("Horizon years", min_value=0.5, max_value=10.0, value=1.0, step=0.5, key="mi_sim_horizon_years"))
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
                            risk_free_rate=float(risk_free_rate_pct) / 100.0,
                            strict=bool(strict_mode),
                            initial_capital=float(initial_capital),
                        )
                        st.session_state["mi_portfolio_sim_result"] = result
                    except Exception as e:
                        st.error(f"Simulation failed: {e}")

            result = st.session_state.get("mi_portfolio_sim_result")
            if export_clicked:
                if not result:
                    st.warning("Run a simulation first, then export.")
                else:
                    try:
                        artifact = generate_decision_brief(
                            simulation_result=result,
                            output_dir=RUN_ARTIFACTS_DIR,
                            format="html",
                            title="Portfolio Decision Brief",
                        )
                        st.success(
                            "Decision Brief exported. "
                            f"HTML: {artifact.get('html_path')} | JSON: {artifact.get('json_path')}"
                        )
                    except Exception as e:
                        st.error(f"Decision Brief export failed: {e}")

            if result:
                ts = result.get("timeseries", {})
                dates = pd.to_datetime(ts.get("dates", []), errors="coerce")
                pvals = pd.Series(ts.get("portfolio_value", []), index=dates, name="Portfolio")
                dds = pd.Series(ts.get("drawdown", []), index=dates, name="Drawdown")
                bvals_raw = ts.get("benchmark_value")
                bvals = pd.Series(bvals_raw, index=dates, name="Benchmark") if bvals_raw is not None else None

                show_risk_overlay = st.checkbox(
                    "Show Risk Overlay (adds a line showing calmer vs riskier periods)",
                    value=False,
                    key="mi_sim_show_risk_overlay",
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
                        risk_df2, _risk_err2 = read_parquet_safe(RISK_CACHE_PATH)
                        if risk_df2 is not None and not risk_df2.empty and {"Date", "RiskScore"}.issubset(set(risk_df2.columns)):
                            rtmp = risk_df2.copy()
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
                            growth_chart = (
                                alt.layer(growth_lines, risk_line)
                                .resolve_scale(y="independent")
                                .properties(height=320, padding={"left": 18, "right": 12, "top": 8, "bottom": 36})
                                .interactive()
                            )
                        else:
                            growth_chart = growth_lines.properties(
                                height=320, padding={"left": 18, "right": 12, "top": 8, "bottom": 36}
                            ).interactive()
                        st.altair_chart(growth_chart, use_container_width=True)
                    else:
                        st.caption("No growth data available.")
                with c2:
                    st.markdown(
                        '<div class="ii-insights-hdr"><h4>Drawdown</h4>'
                        '<span class="ii-insights-info" data-tooltip="Drawdown shows how far the portfolio is below its previous high at each point in time. 0% means it is at a new high; negative values mean it is still recovering from a prior peak.">i</span>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
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
                            .properties(height=320, padding={"left": 18, "right": 12, "top": 8, "bottom": 36})
                            .interactive()
                        )
                        st.altair_chart(draw_chart, use_container_width=True)
                    else:
                        st.caption("No drawdown data available.")

                scenario = result.get("scenario_results")
                if scenario:
                    st.markdown("#### Monte Carlo Scenario Summary")
                    end_pct = scenario.get("ending_value_percentiles", {})
                    dd_pct = scenario.get("max_drawdown_percentiles", {})
                    mc_rows = [
                        {"Statistic": "Ending Value p05", "Value": f"${float(end_pct.get('p05') or 0.0):,.0f}"},
                        {"Statistic": "Ending Value p50", "Value": f"${float(end_pct.get('p50') or 0.0):,.0f}"},
                        {"Statistic": "Ending Value p95", "Value": f"${float(end_pct.get('p95') or 0.0):,.0f}"},
                        {"Statistic": "Max Drawdown p05", "Value": f"{float(dd_pct.get('p05') or 0.0) * 100:.2f}%"},
                        {"Statistic": "Max Drawdown p50", "Value": f"{float(dd_pct.get('p50') or 0.0) * 100:.2f}%"},
                        {"Statistic": "Max Drawdown p95", "Value": f"{float(dd_pct.get('p95') or 0.0) * 100:.2f}%"},
                        {"Statistic": "Probability of Loss", "Value": f"{float(scenario.get('probability_of_loss') or 0.0) * 100:.2f}%"},
                    ]
                    mc_df = pd.DataFrame(mc_rows, columns=["Statistic", "Value"])
                    _render_sortable_centered_table(mc_df, ["Value"])

                summary = result.get("summary", {}) or {}
                _render_performance_metrics_cards(summary)

    with po:
        st.markdown("### Portfolio Optimization")

        def _latest_saved_holdings_map() -> dict[str, float]:
            out_path = DATA_DIR / "portfolio_suggestions_saved.jsonl"
            if not out_path.exists():
                return {}
            try:
                lines = [ln.strip() for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    return {}
                payload = json.loads(lines[-1])
                holds = payload.get("holdings", [])
                out: dict[str, float] = {}
                if isinstance(holds, list):
                    for h in holds:
                        t = str((h or {}).get("Ticker") or "").strip().upper()
                        w = pd.to_numeric(pd.Series([(h or {}).get("Weight (%)", (h or {}).get("WeightPct"))]), errors="coerce").iloc[0]
                        if t and not pd.isna(w):
                            out[t] = float(w)
                return out
            except Exception:
                return {}

        @st.cache_data(show_spinner=False, ttl=300)
        def _price_cov_for_tickers(tickers: tuple[str, ...], lookback_years: int) -> pd.DataFrame:
            px, _pe = read_parquet_safe(PRICES_CACHE_PATH)
            if px is None or px.empty:
                return pd.DataFrame()
            req = {"Ticker", "Date", "AdjClose"}
            if not req.issubset(set(px.columns)):
                return pd.DataFrame()
            p = px.copy()
            p["Ticker"] = p["Ticker"].astype(str).str.upper().str.strip()
            p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
            p["AdjClose"] = pd.to_numeric(p["AdjClose"], errors="coerce")
            p = p.dropna(subset=["Ticker", "Date", "AdjClose"])
            p = p[p["Ticker"].isin(list(tickers))]
            if p.empty:
                return pd.DataFrame()
            end_date = p["Date"].max()
            start_date = end_date - pd.Timedelta(days=int(365 * lookback_years))
            p = p[p["Date"] >= start_date]
            wide = p.pivot_table(index="Date", columns="Ticker", values="AdjClose", aggfunc="last").sort_index()
            rets = wide.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
            if rets.empty:
                return pd.DataFrame()
            cov = rets.cov().fillna(0.0) * 252.0
            return cov

        def _bounded_weights(raw: np.ndarray, min_w: float, max_w: float) -> np.ndarray:
            n = len(raw)
            if n == 0:
                return raw
            x = np.array(raw, dtype=float)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.maximum(x, 0.0)
            if x.sum() <= 0:
                x = np.repeat(1.0 / n, n)
            else:
                x = x / x.sum()

            min_w = max(0.0, float(min_w))
            max_w = min(1.0, float(max_w))
            if n * min_w > 1.0:
                min_w = 1.0 / n
            if max_w < min_w:
                max_w = min_w

            for _ in range(25):
                x = np.clip(x, min_w, max_w)
                s = float(x.sum())
                if s <= 0:
                    x = np.repeat(1.0 / n, n)
                    break
                x = x / s
                if np.all(x >= (min_w - 1e-9)) and np.all(x <= (max_w + 1e-9)):
                    break
            return x

        def _enforce_sector_cap(weights: np.ndarray, sectors: list[str], cap: float) -> np.ndarray:
            x = np.array(weights, dtype=float)
            if len(x) == 0:
                return x
            cap = float(cap)
            for _ in range(10):
                sec_sum: dict[str, float] = {}
                for w, s in zip(x, sectors):
                    sec_sum[s] = sec_sum.get(s, 0.0) + float(w)
                offenders = [s for s, v in sec_sum.items() if v > cap + 1e-9]
                if not offenders:
                    break
                excess = 0.0
                locked = np.zeros(len(x), dtype=bool)
                for s in offenders:
                    idx = np.array([i for i, ss in enumerate(sectors) if ss == s], dtype=int)
                    total = float(x[idx].sum())
                    if total <= 0:
                        continue
                    target = cap
                    factor = target / total
                    x[idx] = x[idx] * factor
                    excess += total - target
                    locked[idx] = True
                free = ~locked
                if free.any() and excess > 0:
                    base = x[free]
                    if base.sum() <= 0:
                        x[free] += excess / int(free.sum())
                    else:
                        x[free] += excess * (base / base.sum())
                s_all = float(x.sum())
                if s_all > 0:
                    x = x / s_all
            return x

        base = _build_portfolio_suggestion_base()
        if base is None or base.empty:
            st.caption("Optimization unavailable because source data is missing.")
        else:
            df = base.copy()
            df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
            df["QualityScore"] = pd.to_numeric(df["QualityScore"], errors="coerce")
            df["QualityTier"] = df["QualityTier"].astype(str)
            if "Sector" not in df.columns:
                df["Sector"] = "Unknown"
            df["Sector"] = df["Sector"].fillna("Unknown").astype(str)
            df = df.dropna(subset=["Ticker", "QualityScore"])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                objective = st.selectbox("Objective", ["Max Sharpe", "Min Volatility", "Max Quality"], index=0, key="mi_opt_objective")
            with c2:
                lookback = int(st.selectbox("Lookback years", [1, 3, 5], index=1, key="mi_opt_lookback"))
            with c3:
                min_w_pct = float(st.slider("Min weight (%)", 0.0, 10.0, 0.0, 0.5, key="mi_opt_minw"))
            with c4:
                max_w_pct = float(st.slider("Max weight (%)", 5.0, 40.0, 15.0, 0.5, key="mi_opt_maxw"))

            d1, d2, d3 = st.columns(3)
            with d1:
                sector_cap_pct = float(st.slider("Max sector (%)", 10.0, 60.0, 30.0, 1.0, key="mi_opt_sector_cap"))
            with d2:
                max_names = int(st.selectbox("Max holdings", [8, 10, 12, 15, 20], index=1, key="mi_opt_max_names"))
            with d3:
                use_saved = st.checkbox("Start from saved portfolio universe", value=True, key="mi_opt_use_saved")

            saved_map = _latest_saved_holdings_map()
            if use_saved and saved_map:
                universe_tickers = set(saved_map.keys())
                pool = df[df["Ticker"].isin(universe_tickers)].copy()
            else:
                pool = df.copy()

            pool = pool.sort_values("QualityScore", ascending=False).head(max_names).copy()
            if pool.empty:
                st.caption("No candidates available after filters.")
            else:
                tks = tuple(pool["Ticker"].astype(str).tolist())
                cov = _price_cov_for_tickers(tks, lookback)
                vol_map: dict[str, float] = {}
                if cov is not None and not cov.empty:
                    for t in tks:
                        if t in cov.index:
                            v = float(max(cov.loc[t, t], 1e-8))
                            vol_map[t] = float(np.sqrt(v))
                for t in tks:
                    vol_map.setdefault(t, 0.25)

                qscore = pd.to_numeric(pool["QualityScore"], errors="coerce").fillna(pool["QualityScore"].median()).values.astype(float)
                qz = (qscore - float(np.mean(qscore))) / (float(np.std(qscore)) + 1e-9)
                mu = 0.06 + 0.08 * qz
                vols = np.array([max(0.05, vol_map.get(t, 0.25)) for t in tks], dtype=float)

                if objective == "Min Volatility":
                    raw = 1.0 / np.maximum(vols, 1e-6)
                elif objective == "Max Quality":
                    raw = np.maximum(qscore, 0.0)
                else:
                    raw = np.maximum(mu / np.maximum(vols, 1e-6), 0.0)

                w = _bounded_weights(raw, min_w_pct / 100.0, max_w_pct / 100.0)
                w = _enforce_sector_cap(w, pool["Sector"].astype(str).tolist(), sector_cap_pct / 100.0)
                w = _bounded_weights(w, min_w_pct / 100.0, max_w_pct / 100.0)

                cur = np.array([saved_map.get(t, 0.0) / 100.0 for t in tks], dtype=float)
                opt = np.round(w * 100.0, 1)
                cur_pct = np.round(cur * 100.0, 1)
                delta = np.round(opt - cur_pct, 1)

                reasons = []
                for i, t in enumerate(tks):
                    if objective == "Min Volatility":
                        reasons.append("Lower historical volatility contribution.")
                    elif objective == "Max Quality":
                        reasons.append("Higher quality score relative to peers.")
                    else:
                        reasons.append("Better quality-to-risk tradeoff.")

                out = pool[["Ticker", "Sector", "QualityTier"]].copy()
                out["Current (%)"] = cur_pct
                out["Optimized (%)"] = opt
                out["Delta (%)"] = delta
                out["Reason"] = reasons
                out = out.rename(columns={"QualityTier": "Tier"})
                _render_sortable_centered_table(out, ["Current (%)", "Optimized (%)", "Delta (%)", "Tier"], page_size=10)

                if st.button("Save Optimized Portfolio", key="mi_save_optimized_portfolio", use_container_width=False):
                    out_path = DATA_DIR / "portfolio_optimized_saved.jsonl"
                    payload = {
                        "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
                        "objective": objective,
                        "lookback_years": lookback,
                        "min_weight_pct": min_w_pct,
                        "max_weight_pct": max_w_pct,
                        "sector_cap_pct": sector_cap_pct,
                        "holdings": [
                            {"Ticker": str(t), "Weight (%)": float(wp)}
                            for t, wp in zip(tks, opt)
                        ],
                    }
                    try:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload) + "\n")
                        st.success(f"Saved to {out_path}")
                    except Exception as e:
                        st.error(f"Could not save optimized portfolio: {e}")


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

    valid_sim_tickers: set[str] = set()
    prices_df, _pxerr = read_parquet_safe(PRICES_CACHE_PATH)
    if prices_df is not None and not prices_df.empty and "Ticker" in prices_df.columns:
        valid_sim_tickers = set(prices_df["Ticker"].astype(str).str.upper().str.strip().tolist())

    fundamentals = _load_fundamentals_union()
    if st.session_state.get("sim_force_manual_weighting_next", False):
        st.session_state["sim_weighting_mode"] = "Manual weights"
        st.session_state["sim_force_manual_weighting_next"] = False

    if "sim_holdings_rows" not in st.session_state:
        st.session_state["sim_holdings_rows"] = _default_sim_holdings_rows()
    st.session_state["sim_holdings_rows"] = _clean_sim_holdings_rows(st.session_state.get("sim_holdings_rows"))
    sim_rows = st.session_state["sim_holdings_rows"]
    auto_rf, auto_src = _auto_risk_free_rate_pct()
    if "sim_risk_free_rate_pct" not in st.session_state:
        st.session_state["sim_risk_free_rate_pct"] = float(auto_rf)
        st.session_state["sim_risk_free_rate_source"] = auto_src
    else:
        # Recover from stale fallback labels when treasury cache becomes available.
        prior_src = str(st.session_state.get("sim_risk_free_rate_source", ""))
        if prior_src.lower().startswith("fallback default") and auto_src.lower().startswith("auto from treasury"):
            st.session_state["sim_risk_free_rate_pct"] = float(auto_rf)
            st.session_state["sim_risk_free_rate_source"] = auto_src

    left_group, right_group = st.columns([2.4, 2.4], vertical_alignment="top")
    with right_group:
        top_a, top_b = st.columns(2, vertical_alignment="bottom")
        with top_a:
            weighting_mode = st.selectbox(
                "Weighting",
                ["Equal weight", "Market cap weight", "Manual weights"],
                index=0,
                key="sim_weighting_mode",
            )
        with top_b:
            benchmark = st.selectbox(
                "Benchmark",
                SIM_BENCHMARK_OPTIONS,
                index=0,
                key="sim_benchmark",
            )

        mid_a, mid_b = st.columns(2, vertical_alignment="bottom")
        with mid_a:
            lookback_years = st.selectbox("Lookback period", [1, 3, 5, 10], index=2, key="sim_lookback")
        with mid_b:
            rebalance_label = st.selectbox("Rebalance", ["None", "Monthly"], index=0, key="sim_rebalance")

        bot_a, bot_b, bot_c, bot_d = st.columns([1.05, 1.20, 1.10, 1.10], vertical_alignment="bottom")
        with bot_a:
            mode_label = st.selectbox("Simulation mode", ["Historical", "Monte Carlo"], index=0, key="sim_mode")
        with bot_b:
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
        with bot_c:
            risk_free_rate_pct = float(
                st.number_input(
                    "Risk-free rate (%)",
                    min_value=0.0,
                    max_value=15.0,
                    step=0.1,
                    key="sim_risk_free_rate_pct",
                )
            )
        with bot_d:
            strict_mode = st.checkbox("Strict missing-data mode", value=False, key="sim_strict")
        source_note = str(st.session_state.get("sim_risk_free_rate_source", "Fallback default (treasury cache unavailable)"))
        st.caption(f"Risk-free rate source: {risk_free_rate_pct:.2f}% ({source_note})")

    with left_group:
        st.caption("Portfolio holdings")
        popover_label = "Edit holdings list"
        popover_ctx = (
            st.popover(popover_label, use_container_width=True)
            if hasattr(st, "popover")
            else st.expander(popover_label, expanded=False)
        )
        with popover_ctx:
            st.caption("Enter each ticker and its target portfolio weight (%).")
            editor_df = pd.DataFrame(sim_rows, columns=["Ticker", "WeightPct"])
            editor_kwargs: dict[str, object] = {}
            if hasattr(st, "column_config"):
                editor_kwargs["column_config"] = {
                    "Ticker": st.column_config.TextColumn("Ticker", help="Example: AAPL"),
                    "WeightPct": st.column_config.NumberColumn("Weight (%)", min_value=0.0, step=0.5, format="%.2f"),
                }
            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="sim_holdings_editor",
                **editor_kwargs,
            )
            proposed_rows = _clean_sim_holdings_rows(edited_df.to_dict(orient="records"))
            proposed_total_pct = float(sum(float(r.get("WeightPct", 0.0) or 0.0) for r in proposed_rows))
            over_allocated = proposed_total_pct > 100.0 + 1e-9
            invalid_tickers = sorted(
                [
                    str(r.get("Ticker", "")).upper()
                    for r in proposed_rows
                    if valid_sim_tickers and str(r.get("Ticker", "")).upper() not in valid_sim_tickers
                ]
            )
            if over_allocated:
                st.error(f"Total weight is {proposed_total_pct:.2f}%. Please reduce to 100% or below before applying.")
            if invalid_tickers:
                st.error("Invalid ticker(s): " + ", ".join(invalid_tickers))
            block_apply = over_allocated or bool(invalid_tickers)
            if st.button("Apply holdings", key="sim_apply_holdings", use_container_width=True, disabled=block_apply):
                st.session_state["sim_holdings_rows"] = proposed_rows
                st.session_state["sim_force_manual_weighting_next"] = True
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

        sim_rows = st.session_state.get("sim_holdings_rows", [])
        ticker_text_preview, manual_weights_text_preview, total_pct_preview = _sim_rows_to_inputs(sim_rows)
        tickers_preview = _parse_ticker_input(ticker_text_preview)
        effective_holdings, preview_warnings = _build_holdings(
            tickers=tickers_preview,
            weighting_mode=weighting_mode,
            fundamentals_df=fundamentals,
            manual_weights_text=manual_weights_text_preview,
        )
        st.caption(f"Holdings: {len(sim_rows)} | Total entered: {total_pct_preview:.2f}%")
        if effective_holdings:
            preview_df = pd.DataFrame(
                [{"Ticker": t, "Weight (%)": round(float(w) * 100.0, 2)} for t, w in effective_holdings]
            )
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            if weighting_mode == "Manual weights":
                total_effective = float(preview_df["Weight (%)"].sum()) if not preview_df.empty else 0.0
                st.caption(f"Effective total: {total_effective:.2f}%")
        elif sim_rows:
            preview_df = pd.DataFrame(sim_rows, columns=["Ticker", "WeightPct"]).rename(columns={"WeightPct": "Weight (%)"})
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        if preview_warnings:
            st.caption(preview_warnings[0])
        if not sim_rows:
            st.caption("No holdings entered yet.")

    ticker_text, manual_weights_text, total_weight_pct = _sim_rows_to_inputs(st.session_state.get("sim_holdings_rows", []))
    if weighting_mode == "Manual weights" and ticker_text:
        if abs(total_weight_pct - 100.0) > 0.01:
            st.info(
                f"Manual weights total {total_weight_pct:.2f}%. "
                "The simulator will normalize weights to sum to 100%."
            )

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
                    risk_free_rate=float(risk_free_rate_pct) / 100.0,
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
                output_dir=RUN_ARTIFACTS_DIR,
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
                growth_chart = (
                    alt.layer(growth_lines, risk_line)
                    .resolve_scale(y="independent")
                    .properties(height=320, padding={"left": 18, "right": 12, "top": 8, "bottom": 36})
                    .interactive()
                )
            else:
                growth_chart = growth_lines.properties(
                    height=320, padding={"left": 18, "right": 12, "top": 8, "bottom": 36}
                ).interactive()
            st.altair_chart(growth_chart, use_container_width=True)
        else:
            st.caption("No growth data available.")
    with c2:
        st.markdown(
            '<div class="ii-insights-hdr"><h4>Drawdown</h4>'
            '<span class="ii-insights-info" data-tooltip="Drawdown shows how far the portfolio is below its previous high at each point in time. 0% means it is at a new high; negative values mean it is still recovering from a prior peak. What to look for: deeper drops (for example -30% vs -10%) mean more severe pain, and long periods below 0% mean slower recovery. Use this chart to judge how much downside and recovery time you can comfortably handle.">i</span>'
            "</div>",
            unsafe_allow_html=True,
        )
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
                .properties(height=320, padding={"left": 18, "right": 12, "top": 8, "bottom": 36})
                .interactive()
            )
            st.altair_chart(draw_chart, use_container_width=True)
        else:
            st.caption("No drawdown data available.")

    insights = result.get("decision_insights", [])
    if not insights:
        st.caption("No insights available.")
    else:
        split_idx = (len(insights) + 1) // 2
        left_items = "".join(f"<li>{line}</li>" for line in insights[:split_idx])
        right_items = "".join(f"<li>{line}</li>" for line in insights[split_idx:])
        st.markdown(
            f'<div class="ii-insights"><div class="ii-insights-hdr"><h4>Decision Insights</h4><span class="ii-insights-info" data-tooltip="How to read the last three bullets: (1) Concentration means how much your portfolio depends on a few positions. Moderate means some diversification, but a few names can still move results materially. (2) Volatility means how bumpy returns are year to year; higher volatility can mean larger ups and downs and may feel uncomfortable for conservative risk tolerance. (3) Downside tail risk uses CVaR 95%: it estimates the average loss on very bad days (the worst 5% of days). A value below -3% daily means losses can be meaningfully large when markets are stressed. These are rule-based summaries from your simulation outputs, not investment advice.">i</span></div><div class="ii-insights-cols"><ul>{left_items}</ul><ul>{right_items}</ul></div></div>',
            unsafe_allow_html=True,
        )

    _render_performance_metrics_cards(result.get("summary", {}) or {})

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
    quality_df, _qerr = read_parquet_safe(QUALITY_CACHE_PATH)
    regime_df, _rerr = read_parquet_safe(REGIME_CACHE_PATH)
    risk_df, _risk_err = read_parquet_safe(RISK_CACHE_PATH)

    if quality_df is None and regime_df is None and risk_df is None:
        st.warning("Model artifacts not found. Run pipeline with run_models=True.")
        return

    latest_regime = "Unknown"
    latest_regime_conf = float("nan")
    latest_regime_date = pd.NaT
    if regime_df is not None and not regime_df.empty and {"Date", "RegimeLabel"}.issubset(set(regime_df.columns)):
        rtmp = regime_df.copy()
        rtmp["Date"] = pd.to_datetime(rtmp["Date"], errors="coerce")
        if "ConfidenceScore" in rtmp.columns:
            rtmp["ConfidenceScore"] = pd.to_numeric(rtmp["ConfidenceScore"], errors="coerce")
        rtmp = rtmp.dropna(subset=["Date"]).sort_values("Date")
        if not rtmp.empty:
            last = rtmp.iloc[-1]
            latest_regime = str(last.get("RegimeLabel", "Unknown"))
            latest_regime_conf = float(last.get("ConfidenceScore")) if pd.notna(last.get("ConfidenceScore")) else float("nan")
            latest_regime_date = pd.to_datetime(last.get("Date"), errors="coerce")

    latest_risk = "Unknown"
    latest_risk_score = float("nan")
    latest_risk_date = pd.NaT
    risk_delta = float("nan")
    if risk_df is not None and not risk_df.empty and {"Date", "RiskLevel"}.issubset(set(risk_df.columns)):
        ktmp = risk_df.copy()
        ktmp["Date"] = pd.to_datetime(ktmp["Date"], errors="coerce")
        if "RiskScore" in ktmp.columns:
            ktmp["RiskScore"] = pd.to_numeric(ktmp["RiskScore"], errors="coerce")
        ktmp = ktmp.dropna(subset=["Date"]).sort_values("Date")
        if not ktmp.empty:
            last = ktmp.iloc[-1]
            latest_risk = str(last.get("RiskLevel", "Unknown"))
            latest_risk_score = float(last.get("RiskScore")) if pd.notna(last.get("RiskScore")) else float("nan")
            latest_risk_date = pd.to_datetime(last.get("Date"), errors="coerce")
            if len(ktmp) >= 2 and np.isfinite(latest_risk_score):
                win_start = ktmp["Date"].max() - pd.Timedelta(days=7)
                base = ktmp[ktmp["Date"] <= win_start]
                if base.empty:
                    base = ktmp.head(1)
                base_score = float(base.iloc[-1].get("RiskScore")) if pd.notna(base.iloc[-1].get("RiskScore")) else float("nan")
                if np.isfinite(base_score):
                    risk_delta = float(latest_risk_score - base_score)

    stocks_tracked = int(len(quality_df)) if quality_df is not None else 0
    strong = neutral = weak = 0
    if quality_df is not None and not quality_df.empty and "QualityTier" in quality_df.columns:
        counts = quality_df["QualityTier"].astype(str).value_counts()
        strong = int(counts.get("Strong", 0))
        neutral = int(counts.get("Neutral", 0))
        weak = int(counts.get("Weak", 0))

    as_of = latest_regime_date if pd.notna(latest_regime_date) else latest_risk_date
    as_of_txt = as_of.strftime("%b %d, %Y") if pd.notna(as_of) else "latest available date"

    risk_trend = "unavailable"
    if np.isfinite(risk_delta):
        if risk_delta > 0.25:
            risk_trend = "rising"
        elif risk_delta < -0.25:
            risk_trend = "falling"
        else:
            risk_trend = "stable"

    confidence_level = "Unknown"
    if np.isfinite(latest_regime_conf):
        if latest_regime_conf >= 0.70:
            confidence_level = "High"
        elif latest_regime_conf >= 0.55:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

    action_level = "Low"
    action_text = "Monitor"
    action_class = "diw-status-pill-low"
    if (
        str(latest_risk).lower() == "elevated"
        or (np.isfinite(latest_risk_score) and latest_risk_score >= 65.0)
    ):
        action_level = "High"
        action_text = "Act now"
        action_class = "diw-status-pill-high"
    elif (
        str(latest_risk).lower() == "moderate"
        or (np.isfinite(latest_risk_score) and latest_risk_score >= 35.0)
    ):
        action_level = "Medium"
        action_text = "Review this week"
        action_class = "diw-status-pill-medium"

    risk_score_chip = "n/a"
    if np.isfinite(latest_risk_score):
        risk_score_chip = f"{latest_risk_score:.1f}"
    risk_delta_chip = "n/a"
    if np.isfinite(risk_delta):
        sign = "+" if risk_delta >= 0 else "-"
        risk_delta_chip = f"{sign}{abs(risk_delta):.1f}"

    snapshot_html = f"""
    <div class="diw-wrap">
      <div class="diw-head">
        <div>
          <div class="diw-kicker">Decision Intelligence</div>
          <div class="diw-title">What Should I Do Now?</div>
          <div class="diw-sub">As of {html.escape(as_of_txt)}</div>
        </div>
        <div class="diw-live">Live</div>
      </div>
      <div class="diw-status-strip">
        <span>Market mood <strong>{html.escape(latest_regime)}</strong></span>
        <span class="diw-sep">â€¢</span>
        <span>Risk <strong>{html.escape(latest_risk)}</strong> ({html.escape(risk_trend)})</span>
        <span class="diw-sep">â€¢</span>
        <span class="diw-status-pill {action_class}">Action level: {html.escape(action_level)} - {html.escape(action_text)}</span>
        <span class="diw-sep">â€¢</span>
        <span>{stocks_tracked} stocks tracked</span>
      </div>
      <div class="diw-section-title">What Matters Right Now</div>
      <div class="diw-cards">
        <div class="diw-card diw-card-blue">
          <div class="diw-card-k">Strong Opportunities</div>
          <div class="diw-card-v">{strong}</div>
          <div class="diw-card-note">High-quality names to research for new positions.</div>
        </div>
        <div class="diw-card diw-card-amber">
          <div class="diw-card-k">Weak Names</div>
          <div class="diw-card-v">{weak}</div>
          <div class="diw-card-note">Prioritize review for reduce/exit decisions.</div>
        </div>
        <div class="diw-card diw-card-blue">
          <div class="diw-card-k">Watchlist</div>
          <div class="diw-card-v">{neutral}</div>
          <div class="diw-card-note">No immediate action. Monitor for rating changes.</div>
        </div>
      </div>
      <div class="diw-section-title">Your Next Actions</div>
      <div class="diw-action-grid">
        <div class="diw-action-card">
          <div class="diw-action-title">Start with strong candidates</div>
          <div class="diw-action-copy">In a {html.escape(latest_regime)} market, focus research on the best-positioned names first.</div>
          <div class="diw-action-meta">Confidence: {html.escape(confidence_level)}</div>
          <div class="diw-action-cta">View top candidates</div>
        </div>
        <div class="diw-action-card">
          <div class="diw-action-title">Review weak names you hold</div>
          <div class="diw-action-copy">Risk is {html.escape(risk_trend)}. Re-check thesis and trim where conviction is low.</div>
          <div class="diw-action-meta">Confidence: {html.escape(confidence_level)}</div>
          <div class="diw-action-cta">Review holdings at risk</div>
        </div>
        <div class="diw-action-card">
          <div class="diw-action-title">Check concentration</div>
          <div class="diw-action-copy">When risk trends up, concentration hurts. Rebalance oversized positions.</div>
          <div class="diw-action-meta">Confidence: Medium</div>
          <div class="diw-action-cta">Review allocation</div>
        </div>
        <div class="diw-action-card">
          <div class="diw-action-title">Set a risk alert</div>
          <div class="diw-action-copy">Current risk: {html.escape(risk_score_chip)} ({html.escape(latest_risk)}). Set an alert at 50 (High).</div>
          <div class="diw-action-meta">Weekly change: {html.escape(risk_delta_chip)} points</div>
          <div class="diw-action-cta">Set alert</div>
        </div>
      </div>
    </div>
    """
    st.markdown(snapshot_html, unsafe_allow_html=True)


def _show_diagnostics_tab() -> None:
    st.markdown("## Diagnostics")
    st.caption("Technical model diagnostics and cache-health details.")

    quality_df, _qerr = read_parquet_safe(QUALITY_CACHE_PATH)
    regime_df, _rerr = read_parquet_safe(REGIME_CACHE_PATH)
    risk_df, _risk_err = read_parquet_safe(RISK_CACHE_PATH)

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
            _render_sortable_centered_table(dist, ["Count"])

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

        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
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
        st.caption(f"`{MODEL_REGISTRY_PATH}` not available.")

    st.markdown("#### Model Health")
    try:
        import json

        with open(MODEL_HEALTH_PATH, "r", encoding="utf-8") as f:
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
                live_specs: dict[str, tuple[str, list[str]]] = {
                    "fundamentals_cache": (FUNDAMENTALS_CACHE_PATH, ["Ticker"]),
                    "prices_cache": (PRICES_CACHE_PATH, ["Ticker", "Date", "AdjClose"]),
                    "treasury_yields_cache": (TREASURY_YIELDS_CACHE_PATH, ["Date", "10Y", "2Y", "3M"]),
                }
                for idx, r in cov_df.iterrows():
                    cache_name = str(r.get("cache") or "")
                    if cache_name not in live_specs:
                        continue
                    rel_path, req_cols = live_specs[cache_name]
                    cands = [rel_path]
                    rp = Path(rel_path)
                    if not rp.is_absolute():
                        cands.append(str((ROOT / rp).resolve()))
                    live_status = None
                    chosen_path = rel_path
                    for p in cands:
                        s = get_cache_status(p, 365, required_columns=req_cols)
                        if s.exists:
                            live_status = s
                            chosen_path = p
                            break
                        if live_status is None:
                            live_status = s
                    if live_status is None:
                        continue
                    cov_df.at[idx, "exists"] = bool(live_status.exists)
                    cov_df.at[idx, "schema_ok"] = bool(live_status.schema_ok)
                    cov_df.at[idx, "age_days"] = round(float(live_status.age_days), 2) if live_status.exists else None
                    live_df, _lerr = read_parquet_safe(chosen_path) if live_status.exists else (None, None)
                    cov_df.at[idx, "row_count"] = int(len(live_df)) if live_df is not None else None
                preferred_cov_cols = ["cache", "exists", "schema_ok", "row_count", "status", "value"]
                if "age_days" in cov_df.columns:
                    preferred_cov_cols.insert(4, "age_days")
                cov_cols = [c for c in preferred_cov_cols if c in cov_df.columns]
                if not cov_cols:
                    cov_cols = list(cov_df.columns)
                _render_sortable_centered_table(cov_df[cov_cols], ["exists", "schema_ok"])
    except Exception:
        st.caption(f"`{MODEL_HEALTH_PATH}` not available.")


def _show_explainability_tab() -> None:
    st.markdown("## Explainability")

    import json

    feature_name_map = {
        "Revenue_Growth_YoY_Pct": "Revenue growth",
        "EBITDA_Margin": "Profit margins (EBITDA)",
        "ROE": "Returns on equity (ROE)",
        "FreeCashFlow_Margin": "Free cash flow quality",
        "Volatility_63D_stability": "Stable price behavior (63-day)",
        "Drawdown_252D_stability": "Smaller drawdowns (12-month)",
    }
    positive_explanations = {
        "Revenue_Growth_YoY_Pct": "Sales are growing at a healthy pace.",
        "EBITDA_Margin": "A meaningful share of revenue converts into operating earnings.",
        "ROE": "The company generates strong profit relative to shareholder capital.",
        "FreeCashFlow_Margin": "Cash generation supports reinvestment and resilience.",
        "Volatility_63D_stability": "The stock has not been unusually volatile recently.",
        "Drawdown_252D_stability": "The stock has avoided severe peak-to-trough drops over the past year.",
    }
    negative_explanations = {
        "Revenue_Growth_YoY_Pct": "Growth is soft relative to peers.",
        "EBITDA_Margin": "Margin profile is weaker than stronger peers.",
        "ROE": "Profitability on shareholder capital is lagging.",
        "FreeCashFlow_Margin": "Cash conversion is less supportive than top names.",
        "Volatility_63D_stability": "Recent price swings are elevated.",
        "Drawdown_252D_stability": "The stock has seen a meaningful 12-month drawdown, signaling downside risk.",
    }
    risk_driver_text = {
        "VolatilityExpansion": "Price swings across the market are getting larger, a sign of rising uncertainty.",
        "CorrelationSpike": "Stocks are moving together more often, so diversification protects less than usual.",
        "RapidDrawdown": "Some names are dropping sharply in short windows, signaling fragile sentiment.",
        "YieldCurveInversion": "The yield curve remains inverted, which often aligns with cautious risk conditions.",
        "RateShock": "Interest-rate moves have been abrupt, making valuation and risk pricing less stable.",
    }

    qexp, _qe = read_parquet_safe(QUALITY_EXPLAIN_PATH)
    qbase, _qb = read_parquet_safe(QUALITY_CACHE_PATH)
    revid, _re = read_parquet_safe(REGIME_EVIDENCE_PATH)
    revbase, _rb = read_parquet_safe(REGIME_CACHE_PATH)
    kevid, _ke = read_parquet_safe(RISK_EVIDENCE_PATH)
    kbase, _kb = read_parquet_safe(RISK_CACHE_PATH)

    st.markdown("### 1. Why Is This Stock Rated This Way?")
    src_quality = qexp if qexp is not None and not qexp.empty else qbase
    if src_quality is None or src_quality.empty or "Ticker" not in src_quality.columns:
        st.caption("No stock-level explanation data available.")
    else:
        q = src_quality.copy()
        q["Ticker"] = q["Ticker"].astype(str).str.upper().str.strip()
        tickers = sorted(q["Ticker"].dropna().unique().tolist())
        ticker = st.selectbox("Stock", options=tickers, index=0, key="xpl_ticker")
        row = q[q["Ticker"] == ticker].tail(1).iloc[0]

        score = float(pd.to_numeric(row.get("QualityScore"), errors="coerce")) if pd.notna(row.get("QualityScore")) else float("nan")
        tier = str(row.get("QualityTier", "Unknown"))
        as_of = str(row.get("FeatureAsOfDate", "latest"))
        if (as_of == "latest" or as_of.lower() == "unknown") and "Date" in q.columns:
            dtmp = pd.to_datetime(q[q["Ticker"] == ticker]["Date"], errors="coerce").dropna()
            if not dtmp.empty:
                as_of = dtmp.max().strftime("%b %d, %Y")
        score_txt = f"{score:.1f}" if np.isfinite(score) else "n/a"

        marker = 50.0 if not np.isfinite(score) else max(0.0, min(100.0, score))
        st.markdown(
            f"""
            <div class="diw-breakdown" style="grid-template-columns: 1fr; gap: 10px;">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
                <div>
                  <div class="diw-title" style="font-size:1.8rem;">{html.escape(ticker)}</div>
                  <div class="diw-muted">As of {html.escape(as_of)} - <span class="diw-chip">{html.escape(tier)}</span></div>
                </div>
                <div style="text-align:right;">
                  <div class="diw-title" style="font-size:2rem;">{html.escape(score_txt)}</div>
                  <div class="diw-muted">Quality score / 100</div>
                </div>
              </div>
              <div style="position:relative; height:10px; border-radius:999px; background:linear-gradient(90deg,#ff5e5e 0%,#f1b434 50%,#21c896 100%);">
                <div style="position:absolute; left:calc({marker}% - 2px); top:-6px; width:4px; height:22px; background:#f3f6fd; border-radius:4px;"></div>
              </div>
              <div style="display:flex; justify-content:space-between;" class="diw-muted">
                <span>Weak</span><span>Strong</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pos_raw = [x.strip() for x in str(row.get("TopPositiveDrivers", "None")).split(",") if x.strip() and x.strip().lower() != "none"]
        neg_raw = [x.strip() for x in str(row.get("TopNegativeDrivers", "None")).split(",") if x.strip() and x.strip().lower() != "none"]

        contrib = {}
        if "ContributionJSON" in row:
            try:
                contrib = json.loads(str(row.get("ContributionJSON", "{}")))
            except Exception:
                contrib = {}
        if not pos_raw and contrib:
            pos_raw = [k for k, v in sorted(contrib.items(), key=lambda kv: float(kv[1]), reverse=True) if float(v) > 0][:3]
        if not neg_raw and contrib:
            neg_raw = [k for k, v in sorted(contrib.items(), key=lambda kv: float(kv[1])) if float(v) < 0][:3]

        st.markdown("#### What's Helping This Stock")
        if not pos_raw:
            st.caption("No strong positive drivers were detected.")
        else:
            for feat in pos_raw[:3]:
                label = feature_name_map.get(feat, feat.replace("_", " "))
                expl = positive_explanations.get(feat, "This factor is currently supporting the score.")
                st.markdown(f"- **{label}**: {expl}")

        st.markdown("#### What's Holding It Back")
        if not neg_raw:
            st.caption("No major negative drivers were detected.")
        else:
            for feat in neg_raw[:3]:
                label = feature_name_map.get(feat, feat.replace("_", " "))
                expl = negative_explanations.get(feat, "This factor is currently limiting the score.")
                st.markdown(f"- **{label}**: {expl}")

    st.markdown("### 2. What Is The Market Doing Right Now?")
    src_regime = revid if revid is not None and not revid.empty else revbase
    if src_regime is None or src_regime.empty:
        st.caption("No market-regime data available.")
    else:
        r = src_regime.copy()
        r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
        r = r.dropna(subset=["Date"]).sort_values("Date")
        rr = r.tail(1).iloc[0]
        mood = str(rr.get("RegimeLabel", "Unknown"))
        conf = float(pd.to_numeric(rr.get("ConfidenceScore"), errors="coerce")) if "ConfidenceScore" in rr else float("nan")
        conf_pct = int(round(conf * 100)) if np.isfinite(conf) else None
        if not np.isfinite(conf):
            conf_txt = "Confidence unavailable"
        elif conf >= 0.70:
            conf_txt = f"High confidence ({conf_pct}%)"
        elif conf >= 0.55:
            conf_txt = f"Moderate confidence ({conf_pct}%) - treat as context, not a hard call"
        else:
            conf_txt = f"Low confidence ({conf_pct}%) - avoid large one-shot moves"
        updated = rr["Date"].strftime("%b %d, %Y")

        explanation = {
            "Risk On": "The backdrop is supportive for risk assets. Favor adding in measured steps rather than all at once.",
            "Risk Off": "Conditions are defensive. Focus on risk control, position sizing, and downside protection.",
            "Neutral": "Neither strong risk-on nor risk-off conditions are present. A wait-and-see posture is appropriate.",
        }.get(mood, "Market signal is mixed. Use your portfolio rules and avoid overreacting to one data point.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Market mood**\n\n### {mood}")
        with c2:
            st.markdown(f"**Signal confidence**\n\n### {conf_txt}")
        st.caption(explanation)
        st.caption(f"Updated {updated}")

    st.markdown("### 3. How Risky Is The Overall Environment?")
    src_risk = kevid if kevid is not None and not kevid.empty else kbase
    if src_risk is None or src_risk.empty:
        st.caption("No market-risk data available.")
    else:
        k = src_risk.copy()
        k["Date"] = pd.to_datetime(k["Date"], errors="coerce")
        k = k.dropna(subset=["Date"]).sort_values("Date")
        kr = k.tail(1).iloc[0]

        risk_score = float(pd.to_numeric(kr.get("RiskScore"), errors="coerce")) if "RiskScore" in kr else float("nan")
        risk_level = str(kr.get("RiskLevel", "Unknown"))
        updated = kr["Date"].strftime("%b %d, %Y")
        risk_score_txt = f"{risk_score:.1f} / 100" if np.isfinite(risk_score) else "n/a"
        st.markdown(f"### {risk_level} risk  -  {risk_score_txt}")
        st.caption(f"Updated {updated}")

        evmap = {}
        if "EvidencePointsJSON" in kr:
            try:
                evmap = json.loads(str(kr.get("EvidencePointsJSON", "{}")))
            except Exception:
                evmap = {}
        drivers_order = ["VolatilityExpansion", "CorrelationSpike", "RapidDrawdown", "YieldCurveInversion", "RateShock"]
        shown = 0
        st.markdown("#### What's Driving The Risk")
        for key in drivers_order:
            val = evmap.get(key)
            if isinstance(val, bool):
                active = val
            else:
                try:
                    active = float(val) > 0
                except Exception:
                    active = False
            if not active:
                continue
            shown += 1
            label = {
                "VolatilityExpansion": "Volatility expanding",
                "CorrelationSpike": "Stocks moving together",
                "RapidDrawdown": "Fast drawdowns occurring",
                "YieldCurveInversion": "Yield curve inverted",
                "RateShock": "Rate shocks",
            }.get(key, key)
            st.markdown(f"- **{label}**: {risk_driver_text.get(key, 'This factor is contributing to current risk conditions.')}")
            if shown >= 3:
                break
        if shown == 0:
            st.caption("No dominant risk driver is currently standing out.")

        if "RiskScore" in k.columns:
            trend = k[["Date", "RiskScore"]].copy()
            trend["RiskScore"] = pd.to_numeric(trend["RiskScore"], errors="coerce")
            trend = trend.dropna(subset=["RiskScore"]).sort_values("Date")
            if not trend.empty:
                trend = trend[trend["Date"] >= (trend["Date"].max() - pd.Timedelta(days=183))]
                st.markdown("#### Risk Score Trend (Last 6 Months)")
                ch = (
                    alt.Chart(trend)
                    .mark_line(color="#ffb64b", strokeWidth=3)
                    .encode(
                        x=alt.X("Date:T", title=None),
                        y=alt.Y("RiskScore:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                        tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("RiskScore:Q", title="Risk", format=".1f")],
                    )
                    .properties(height=180)
                )
                st.altair_chart(ch, use_container_width=True)


def _show_uncertainty_tab() -> None:
    st.markdown("## Uncertainty and Confidence")

    qu, _qe = read_parquet_safe(QUALITY_UNCERTAINTY_PATH)
    qbase, _qb = read_parquet_safe(QUALITY_CACHE_PATH)
    rp, _re = read_parquet_safe(REGIME_PROB_PATH)
    rbase, _rb = read_parquet_safe(REGIME_CACHE_PATH)
    ru, _ru = read_parquet_safe(RISK_UNCERTAINTY_PATH)
    kbase, _kb = read_parquet_safe(RISK_CACHE_PATH)

    def _confidence_badge(value: float) -> tuple[str, str]:
        if not np.isfinite(value):
            return "Uncertain", "#ede6d6"
        if value >= 0.80:
            return "Very confident", "#dff4ec"
        if value >= 0.60:
            return "Moderate", "#f3ecd9"
        return "Less certain", "#f3e2dd"

    # 1) Stock confidence
    st.markdown("### How Confident Is The Model About This Stock?")
    src_q = qu if qu is not None and not qu.empty else qbase
    if src_q is None or src_q.empty or "Ticker" not in src_q.columns:
        st.caption("No stock uncertainty data available.")
    else:
        q = src_q.copy()
        q["Ticker"] = q["Ticker"].astype(str).str.upper().str.strip()
        tickers = sorted(q["Ticker"].dropna().unique().tolist())
        t = st.selectbox("Stock", options=tickers, index=0, key="unc_ticker")
        row = q[q["Ticker"] == t].tail(1).iloc[0]

        point = float(pd.to_numeric(row.get("ScoreP50"), errors="coerce")) if "ScoreP50" in row else float(pd.to_numeric(row.get("QualityScore"), errors="coerce"))
        p10 = float(pd.to_numeric(row.get("ScoreP10"), errors="coerce")) if "ScoreP10" in row else float("nan")
        p90 = float(pd.to_numeric(row.get("ScoreP90"), errors="coerce")) if "ScoreP90" in row else float("nan")
        tier = str(row.get("TierMostLikely", row.get("QualityTier", "Neutral")))
        stability = float(pd.to_numeric(row.get("TierStability"), errors="coerce")) if "TierStability" in row else float("nan")
        confidence_base = stability if np.isfinite(stability) else (1.0 - min(1.0, ((p90 - p10) / 20.0 if np.isfinite(p10) and np.isfinite(p90) else 0.5)))
        conf_lbl, conf_bg = _confidence_badge(confidence_base)

        marker = max(48.0, min(58.0, point if np.isfinite(point) else 53.0))
        low = p10 if np.isfinite(p10) else max(48.0, marker - 1.0)
        high = p90 if np.isfinite(p90) else min(58.0, marker + 1.0)
        swing = (high - low) / 2.0
        stable_txt = f"{int(round(max(0, min(100, confidence_base * 100))))}%"
        st.markdown(
            f"""
            <div class="diw-breakdown" style="grid-template-columns: 1fr; gap: 10px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div>
                  <div class="diw-title" style="font-size:1.8rem;">{html.escape(t)} <span class="diw-chip">{html.escape(tier)}</span></div>
                  <div class="diw-muted">Quality score: {point:.1f} / 100</div>
                </div>
                <div style="padding:4px 12px; border-radius:999px; background:{conf_bg}; color:#1b4d3f; font-weight:700;">{conf_lbl}</div>
              </div>
              <div class="diw-muted" style="font-weight:700;">SCORE RANGE UNDER DIFFERENT SCENARIOS</div>
              <div style="position:relative; height:14px; border-radius:999px; background:#1b2230;">
                <div style="position:absolute; left:{((low-48)/10)*100:.1f}%; width:{max(3.0, ((high-low)/10)*100):.1f}%; top:0; bottom:0; border-radius:999px; background:#2f5e8d;"></div>
                <div style="position:absolute; left:{((marker-48)/10)*100:.1f}%; top:-4px; width:10px; height:22px; border-radius:8px; background:#edf3ff;"></div>
              </div>
              <div style="display:flex; justify-content:space-between;" class="diw-muted"><span>48</span><span>50</span><span>52</span><span>54</span><span>56</span><span>58</span></div>
              <div class="diw-muted">Likely range ({low:.1f} - {high:.1f})  ·  Most likely score ({marker:.1f})</div>
              <div style="border-top:1px solid rgba(135,162,196,0.24); padding-top:10px; color:#d1def4;">
                Even in weaker simulated scenarios, the score stays close to the current reading. The <b>{html.escape(tier)}</b> rating appears relatively stable.
              </div>
              <div style="display:flex; gap:10px;">
                <div style="padding:8px 12px; border:1px solid rgba(135,162,196,0.30); border-radius:10px; background:rgba(10,22,41,0.45);">Rating stability <b>{stable_txt}</b></div>
                <div style="padding:8px 12px; border:1px solid rgba(135,162,196,0.30); border-radius:10px; background:rgba(10,22,41,0.45);">Score swing <b>±{swing:.1f} pts</b></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2) Market mood confidence
    st.markdown("### How Sure Is The Model About The Market Mood?")
    src_r = rp if rp is not None and not rp.empty else rbase
    if src_r is None or src_r.empty:
        st.caption("No regime probability data available.")
    else:
        r = src_r.copy()
        r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
        r = r.dropna(subset=["Date"]).sort_values("Date")
        row = r.tail(1).iloc[0]

        p_on = float(pd.to_numeric(row.get("P_RiskOn"), errors="coerce")) if "P_RiskOn" in row else float("nan")
        p_neu = float(pd.to_numeric(row.get("P_Neutral"), errors="coerce")) if "P_Neutral" in row else float("nan")
        p_off = float(pd.to_numeric(row.get("P_RiskOff"), errors="coerce")) if "P_RiskOff" in row else float("nan")
        if not (np.isfinite(p_on) and np.isfinite(p_neu) and np.isfinite(p_off)):
            conf = float(pd.to_numeric(row.get("ConfidenceScore"), errors="coerce"))
            mood = str(row.get("RegimeLabel", "Neutral"))
            p_on = conf if mood == "Risk On" and np.isfinite(conf) else 0.2
            p_neu = conf if mood == "Neutral" and np.isfinite(conf) else 0.5
            p_off = conf if mood == "Risk Off" and np.isfinite(conf) else 0.3
        probs = {"Risk-on": max(0.0, p_on), "Neutral": max(0.0, p_neu), "Risk-off": max(0.0, p_off)}
        mood = max(probs, key=probs.get)
        conf_val = probs[mood]
        conf_lbl, conf_bg = _confidence_badge(conf_val)

        # consecutive stable days
        streak = 1
        if "RegimeLabel" in r.columns and len(r) > 1:
            labels = r["RegimeLabel"].astype(str).tolist()
            last = labels[-1]
            for lbl in reversed(labels[:-1]):
                if lbl == last:
                    streak += 1
                else:
                    break

        st.markdown(
            f"""
            <div class="diw-breakdown" style="grid-template-columns: 1fr; gap: 10px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div>
                  <div class="diw-title" style="font-size:1.9rem;">Market mood: {html.escape(mood)}</div>
                  <div class="diw-muted">The model sees {'no clear bullish or bearish signal right now' if mood=='Neutral' else ('risk-on bias' if mood=='Risk-on' else 'risk-off pressure')}.</div>
                </div>
                <div style="padding:4px 12px; border-radius:999px; background:{conf_bg}; color:#5c3f12; font-weight:700;">{conf_lbl}</div>
              </div>
              <div class="diw-muted" style="font-weight:700;">PROBABILITY OF EACH SCENARIO</div>
              <div style="display:grid; grid-template-columns: 120px 1fr 70px; gap:10px; align-items:center;">
                <div>Neutral</div><div style="height:12px;background:#1b2230;border-radius:999px;"><div style="height:12px;width:{probs['Neutral']*100:.1f}%;background:#4b9bff;border-radius:999px;"></div></div><div style="text-align:right;">{probs['Neutral']*100:.0f}%</div>
                <div>Risk-off</div><div style="height:12px;background:#1b2230;border-radius:999px;"><div style="height:12px;width:{probs['Risk-off']*100:.1f}%;background:#d65a5a;border-radius:999px;"></div></div><div style="text-align:right;">{probs['Risk-off']*100:.0f}%</div>
                <div>Risk-on</div><div style="height:12px;background:#1b2230;border-radius:999px;"><div style="height:12px;width:{probs['Risk-on']*100:.1f}%;background:#2aa37a;border-radius:999px;"></div></div><div style="text-align:right;">{probs['Risk-on']*100:.0f}%</div>
              </div>
              <div style="border-top:1px solid rgba(135,162,196,0.24); padding-top:10px; color:#d1def4;">
                Confidence is {conf_lbl.lower()} at {conf_val*100:.0f}% - meaning the model leans {html.escape(mood)} but still allows for alternative scenarios.
              </div>
              <div style="padding:8px 12px; display:inline-block; border:1px solid rgba(135,162,196,0.30); border-radius:10px; background:rgba(10,22,41,0.45);">Mood stable for <b>{streak}</b> days</div>
              <div class="diw-muted">Updated {row['Date'].strftime('%b %d, %Y')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3) Risk confidence
    st.markdown("### How Sure Is The Model About The Risk Level?")
    src_k = ru if ru is not None and not ru.empty else kbase
    if src_k is None or src_k.empty:
        st.caption("No risk uncertainty data available.")
    else:
        k = src_k.copy()
        k["Date"] = pd.to_datetime(k["Date"], errors="coerce")
        k = k.dropna(subset=["Date"]).sort_values("Date")
        row = k.tail(1).iloc[0]
        risk_level = str(row.get("RiskLevelMostLikely", row.get("RiskLevel", "Moderate")))
        r_mid = float(pd.to_numeric(row.get("RiskP50"), errors="coerce")) if "RiskP50" in row else float(pd.to_numeric(row.get("RiskScore"), errors="coerce"))
        r_low = float(pd.to_numeric(row.get("RiskP10"), errors="coerce")) if "RiskP10" in row else max(0.0, r_mid - 4.0)
        r_high = float(pd.to_numeric(row.get("RiskP90"), errors="coerce")) if "RiskP90" in row else min(100.0, r_mid + 4.0)
        r_swing = (r_high - r_low) / 2.0
        r_stability = float(pd.to_numeric(row.get("RiskLevelStability"), errors="coerce")) if "RiskLevelStability" in row else float("nan")
        confidence_base = r_stability if np.isfinite(r_stability) else max(0.0, 1.0 - min(1.0, (r_swing / 10.0)))
        conf_lbl, conf_bg = _confidence_badge(confidence_base)

        axis_min, axis_max = 20.0, 50.0
        marker = max(axis_min, min(axis_max, r_mid if np.isfinite(r_mid) else 35.0))
        low = max(axis_min, min(axis_max, r_low))
        high = max(axis_min, min(axis_max, r_high))
        st.markdown(
            f"""
            <div class="diw-breakdown" style="grid-template-columns: 1fr; gap: 10px;">
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:flex-start;">
                <div>
                  <div class="diw-title" style="font-size:1.9rem;">Market risk: {html.escape(risk_level)}</div>
                  <div class="diw-muted">Risk score most likely around {marker:.1f} / 100</div>
                </div>
                <div style="padding:4px 12px; border-radius:999px; background:{conf_bg}; color:#5c3f12; font-weight:700;">{conf_lbl}</div>
              </div>
              <div class="diw-muted" style="font-weight:700;">RISK SCORE RANGE UNDER DIFFERENT SCENARIOS</div>
              <div style="position:relative; height:14px; border-radius:999px; background:#1b2230;">
                <div style="position:absolute; left:{((low-axis_min)/(axis_max-axis_min))*100:.1f}%; width:{max(3.0, ((high-low)/(axis_max-axis_min))*100):.1f}%; top:0; bottom:0; border-radius:999px; background:#a66f18;"></div>
                <div style="position:absolute; left:{((marker-axis_min)/(axis_max-axis_min))*100:.1f}%; top:-4px; width:10px; height:22px; border-radius:8px; background:#f3d8a5;"></div>
              </div>
              <div style="display:flex; justify-content:space-between;" class="diw-muted"><span>20</span><span>25</span><span>30</span><span>35</span><span>40</span><span>45</span><span>50</span></div>
              <div class="diw-muted">Likely range ({low:.1f} - {high:.1f})  ·  Most likely ({marker:.1f})</div>
              <div style="border-top:1px solid rgba(135,162,196,0.24); padding-top:10px; color:#d1def4;">
                The risk level is {conf_lbl.lower()} than the stock quality rating. In a pessimistic scenario it can move toward the upper end of the range.
              </div>
              <div style="display:flex; gap:10px;">
                <div style="padding:8px 12px; border:1px solid rgba(135,162,196,0.30); border-radius:10px; background:rgba(10,22,41,0.45);">Rating stability <b>{int(round(max(0,min(100,confidence_base*100))))}%</b></div>
                <div style="padding:8px 12px; border:1px solid rgba(135,162,196,0.30); border-radius:10px; background:rgba(10,22,41,0.45);">Score swing <b>±{r_swing:.1f} pts</b></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        trend = k[["Date"]].copy()
        trend["RiskScore"] = pd.to_numeric(k.get("RiskScore"), errors="coerce")
        trend["RiskP10"] = pd.to_numeric(k.get("RiskP10"), errors="coerce")
        trend["RiskP90"] = pd.to_numeric(k.get("RiskP90"), errors="coerce")
        trend = trend.dropna(subset=["Date"]).sort_values("Date")
        trend = trend[trend["Date"] >= (trend["Date"].max() - pd.Timedelta(days=90))]
        if not trend.empty and trend["RiskScore"].notna().any():
            st.markdown("#### Risk Score Trend - Last 90 Days (With Uncertainty Band)")
            line = alt.Chart(trend.dropna(subset=["RiskScore"])).mark_line(color="#d08a1f", strokeWidth=3).encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("RiskScore:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("RiskScore:Q", title="Risk", format=".1f")],
            )
            if trend["RiskP10"].notna().any() and trend["RiskP90"].notna().any():
                band = alt.Chart(trend.dropna(subset=["RiskP10", "RiskP90"])).mark_area(opacity=0.16, color="#d08a1f").encode(
                    x="Date:T",
                    y="RiskP10:Q",
                    y2="RiskP90:Q",
                )
                st.altair_chart((band + line).properties(height=220), use_container_width=True)
            else:
                st.altair_chart(line.properties(height=220), use_container_width=True)
        st.caption(f"Updated {row['Date'].strftime('%b %d, %Y')}")


def _show_monitoring_tab() -> None:
    st.markdown("## Drift, Monitoring, and Early Warning")

    import json

    def _norm_dates(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_localize(None)
            out = out.dropna(subset=["Date"])
        return out

    drift_df, _de = read_parquet_safe(DRIFT_SIGNALS_PATH)
    drift_hist_df, _dh = read_parquet_safe(DRIFT_SIGNALS_HISTORY_PATH)
    alert_df, _ae = read_parquet_safe(ALERT_LOG_PATH)
    regime_df, _re = read_parquet_safe(REGIME_CACHE_PATH)
    risk_df, _rk = read_parquet_safe(RISK_CACHE_PATH)

    drift_df = _norm_dates(drift_df)
    drift_hist_df = _norm_dates(drift_hist_df)
    alert_df = _norm_dates(alert_df)
    regime_df = _norm_dates(regime_df)
    risk_df = _norm_dates(risk_df)

    if drift_df.empty:
        st.warning("Monitoring signals are not available yet.")
        return

    latest_date = pd.Timestamp.utcnow().normalize()
    if "Date" in drift_df.columns and not drift_df["Date"].dropna().empty:
        latest_date = pd.Timestamp(drift_df["Date"].max()).normalize()
    win_90_start = latest_date - pd.Timedelta(days=90)
    win_60_start = latest_date - pd.Timedelta(days=60)

    drift_90 = drift_df[(drift_df["Date"] >= win_90_start) & (drift_df["Date"] <= latest_date)] if "Date" in drift_df.columns else drift_df.copy()
    alerts_90 = alert_df[(alert_df["Date"] >= win_90_start) & (alert_df["Date"] <= latest_date)] if not alert_df.empty and "Date" in alert_df.columns else alert_df.copy()
    regime_60 = regime_df[(regime_df["Date"] >= win_60_start) & (regime_df["Date"] <= latest_date)] if not regime_df.empty and "Date" in regime_df.columns else regime_df.copy()
    risk_60 = risk_df[(risk_df["Date"] >= win_60_start) & (risk_df["Date"] <= latest_date)] if not risk_df.empty and "Date" in risk_df.columns else risk_df.copy()

    worst_level = "Stable"
    if not drift_90.empty and "DriftLevel" in drift_90.columns:
        levels = drift_90["DriftLevel"].astype(str)
        if (levels == "Severe").any():
            worst_level = "Severe"
        elif (levels == "Drift").any():
            worst_level = "Drift"

    critical_count = int((alerts_90["Severity"].astype(str) == "Critical").sum()) if not alerts_90.empty and "Severity" in alerts_90.columns else 0
    warning_count = int((alerts_90["Severity"].astype(str) == "Warning").sum()) if not alerts_90.empty and "Severity" in alerts_90.columns else 0
    info_count = int((alerts_90["Severity"].astype(str) == "Info").sum()) if not alerts_90.empty and "Severity" in alerts_90.columns else 0

    if worst_level == "Severe":
        banner_title = "Model inputs are changing significantly"
        banner_copy = (
            "The data the model relies on has drifted enough to trigger a Severe alert. "
            "Signals are currently unstable. Scores and ratings are still generated, but treat them with extra caution right now."
        )
        banner_bg = "#f3e3df"
        banner_border = "#d7866f"
        banner_txt = "#5d2c1f"
    elif worst_level == "Drift":
        banner_title = "Some model inputs are shifting"
        banner_copy = (
            "Recent market data has moved away from normal ranges. "
            "Signals remain useful, but confidence is lower than usual."
        )
        banner_bg = "#f6eddc"
        banner_border = "#d8a54d"
        banner_txt = "#5c3f12"
    else:
        banner_title = "Model conditions look stable"
        banner_copy = "No material drift warning is active. Signals appear broadly stable."
        banner_bg = "#e1f0e7"
        banner_border = "#71b18d"
        banner_txt = "#18422c"

    st.markdown(
        f"""
        <div style="border:1px solid {banner_border}; background:{banner_bg}; color:{banner_txt}; border-radius:14px; padding:16px 18px; margin:8px 0 14px 0;">
          <div style="font-size:1.9rem; font-weight:800; line-height:1.2;">{html.escape(banner_title)}</div>
          <div style="margin-top:6px; font-size:1.03rem; line-height:1.5;">{html.escape(banner_copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Active Alerts - Last 90 Days")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Critical", critical_count)
    with c2:
        st.metric("Warnings", warning_count)
    with c3:
        st.metric("Info only", info_count)

    flip_rate = float("nan")
    avg_conf = float("nan")
    if not regime_60.empty and "RegimeLabel" in regime_60.columns:
        r60 = regime_60.sort_values("Date").copy()
        flip_rate = float((r60["RegimeLabel"].astype(str) != r60["RegimeLabel"].astype(str).shift(1)).mean()) if len(r60) > 1 else 0.0
        if "ConfidenceScore" in r60.columns:
            avg_conf = float(pd.to_numeric(r60["ConfidenceScore"], errors="coerce").mean())

    risk_changes = 0
    risk_vol = float("nan")
    if not risk_60.empty:
        rk60 = risk_60.sort_values("Date").copy()
        if "RiskLevel" in rk60.columns:
            risk_changes = int((rk60["RiskLevel"].astype(str) != rk60["RiskLevel"].astype(str).shift(1)).sum())
        if "RiskScore" in rk60.columns:
            rk60["RiskScore"] = pd.to_numeric(rk60["RiskScore"], errors="coerce")
            risk_vol = float(rk60["RiskScore"].std(ddof=1)) if rk60["RiskScore"].notna().sum() > 1 else 0.0

    metric_name_map = {
        "BenchmarkVolatility_63d": "Market volatility inputs have drifted",
        "RegimeFlipRate_60d": "Market direction has been flipping frequently",
        "RiskScoreVol_60d": "Risk score has been volatile",
        "QualityProxyDrift": "Stock-level quality scores are shifting",
    }
    drift_rows = drift_90.copy()
    if not drift_rows.empty and "DriftScore" in drift_rows.columns:
        drift_rows["DriftScore"] = pd.to_numeric(drift_rows["DriftScore"], errors="coerce")
        drift_rows = drift_rows.sort_values("DriftScore", ascending=False)

    flagged: list[dict[str, str]] = []

    def _sev_label(level: str) -> str:
        x = str(level).lower()
        if x in {"critical", "severe"}:
            return "Critical"
        if x in {"warning", "drift"}:
            return "Warning"
        return "Info"

    if np.isfinite(flip_rate) and flip_rate >= 0.10:
        sev = "Critical" if flip_rate >= 0.15 else "Warning"
        flagged.append(
            {
                "title": "Market direction has been flipping frequently",
                "sev": sev,
                "body": f"The market mood has changed direction about {flip_rate*100:.0f}% of the time over the last 60 days, higher than normal.",
                "action": "Do not act on single-day regime shifts; wait for multi-day consistency.",
            }
        )
    if np.isfinite(avg_conf) and avg_conf <= 0.60:
        flagged.append(
            {
                "title": "Signals are oscillating without a clear trend",
                "sev": "Warning",
                "body": f"Average confidence has been about {avg_conf*100:.0f}% over the last 60 days, barely above a coin flip.",
                "action": "Weight fundamentals and valuation more heavily until confidence improves.",
            }
        )
    if risk_changes >= 8:
        sev = "Critical" if risk_changes >= 12 else "Warning"
        flagged.append(
            {
                "title": "Risk score has been volatile",
                "sev": sev,
                "body": f"The risk level changed {risk_changes} times in the last 60 days, so the current reading may not stay stable for long.",
                "action": "Check back frequently; avoid treating risk level as static.",
            }
        )
    if np.isfinite(risk_vol) and risk_vol >= 6:
        sev = "Critical" if risk_vol >= 12 else "Warning"
        flagged.append(
            {
                "title": "Market risk conditions are moving quickly",
                "sev": sev,
                "body": "Day-to-day risk readings have become more volatile than normal.",
                "action": "Use smaller position changes and stagger entries/exits.",
            }
        )

    for _, row in drift_rows.head(5).iterrows():
        lvl = str(row.get("DriftLevel", ""))
        if lvl not in {"Severe", "Drift"}:
            continue
        metric = str(row.get("MetricName", "Model input"))
        title = metric_name_map.get(metric, "Model inputs are changing")
        if any(x["title"] == title for x in flagged):
            continue
        flagged.append(
            {
                "title": title,
                "sev": _sev_label(lvl),
                "body": "The data feeding this part of the model looks different from normal conditions, which can reduce score reliability.",
                "action": "Cross-check with your own view of market conditions before making large moves.",
            }
        )

    st.markdown("### What's Being Flagged")
    if not flagged:
        st.caption("No high-priority instability flags right now.")
    else:
        for item in flagged[:6]:
            sev = item["sev"]
            if sev == "Critical":
                dot = "#ef5a5a"
                pill_bg = "#f3e1dd"
                pill_fg = "#8c2f2f"
            elif sev == "Warning":
                dot = "#d99a2f"
                pill_bg = "#f8efd9"
                pill_fg = "#80550e"
            else:
                dot = "#6aa4df"
                pill_bg = "#e4eef9"
                pill_fg = "#1f4b7d"
            st.markdown(
                f"""
                <div style="border:1px solid rgba(135,162,196,0.24); border-radius:12px; padding:12px 14px; margin:8px 0; background:rgba(10,22,41,0.45);">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <span style="width:11px; height:11px; border-radius:999px; background:{dot}; display:inline-block;"></span>
                    <span style="font-size:1.35rem; font-weight:800; color:#edf4ff;">{html.escape(item['title'])}</span>
                    <span style="margin-left:auto; padding:2px 10px; border-radius:999px; background:{pill_bg}; color:{pill_fg}; font-weight:700; font-size:0.92rem;">{sev}</span>
                  </div>
                  <div style="margin-top:6px; color:#d1def4; line-height:1.45;">{html.escape(item['body'])}</div>
                  <div style="margin-top:5px; color:#b8cae8; font-style:italic;">-> {html.escape(item['action'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### How Stable Are Signals Right Now?")
    m_status = "Stable"
    m_text = "Direction changes are within normal range."
    m_color = "#63c38f"
    if np.isfinite(flip_rate):
        if flip_rate >= 0.15:
            m_status = "Unstable"
            m_color = "#ef5a5a"
        elif flip_rate >= 0.10:
            m_status = "Moderate"
            m_color = "#d99a2f"
        m_text = f"Flipping direction ~{flip_rate*100:.0f}% of the time - {'higher than normal' if flip_rate >= 0.10 else 'within normal range'}."

    r_status = "Stable"
    r_text = "Risk level has been relatively steady."
    r_color = "#63c38f"
    if risk_changes >= 10:
        r_status = "Moderate"
        r_color = "#d99a2f"
        r_text = f"Changed {risk_changes} times in 60 days."
    if risk_changes >= 14:
        r_status = "Unstable"
        r_color = "#ef5a5a"
        r_text = f"Changed {risk_changes} times in 60 days - very unstable."

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="border:1px solid rgba(135,162,196,0.24); border-radius:12px; padding:12px 14px; background:rgba(10,22,41,0.45);">
              <div class="diw-muted">Market mood stability</div>
              <div style="font-size:1.9rem; font-weight:800; color:{m_color}; margin-top:2px;">{m_status}</div>
              <div style="height:7px; border-radius:999px; background:rgba(132,154,184,0.25); margin:8px 0 9px 0;">
                <div style="height:7px; border-radius:999px; background:{m_color}; width:{min(95, max(12, (flip_rate*100 if np.isfinite(flip_rate) else 8)))}%;"></div>
              </div>
              <div class="diw-muted">{html.escape(m_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        bar_w = min(95, max(12, risk_changes * 6 if risk_changes > 0 else 10))
        st.markdown(
            f"""
            <div style="border:1px solid rgba(135,162,196,0.24); border-radius:12px; padding:12px 14px; background:rgba(10,22,41,0.45);">
              <div class="diw-muted">Risk level stability</div>
              <div style="font-size:1.9rem; font-weight:800; color:{r_color}; margin-top:2px;">{r_status}</div>
              <div style="height:7px; border-radius:999px; background:rgba(132,154,184,0.25); margin:8px 0 9px 0;">
                <div style="height:7px; border-radius:999px; background:{r_color}; width:{bar_w}%;"></div>
              </div>
              <div class="diw-muted">{html.escape(r_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Drift Trend - Last 90 Days")
    trend_src = drift_hist_df if not drift_hist_df.empty else drift_df
    if not trend_src.empty and {"Date", "DriftScore"}.issubset(set(trend_src.columns)):
        t = trend_src.copy()
        t["DriftScore"] = pd.to_numeric(t["DriftScore"], errors="coerce")
        t = t.dropna(subset=["Date", "DriftScore"])
        t = t[(t["Date"] >= win_90_start) & (t["Date"] <= latest_date)]
        if not t.empty:
            t["AsOfDate"] = t["Date"].dt.normalize()
            td = t.groupby("AsOfDate", as_index=False)["DriftScore"].max()
            td = td.sort_values("AsOfDate")
            threshold = 0.25
            line = (
                alt.Chart(td)
                .mark_line(color="#ef5a5a", strokeWidth=3)
                .encode(
                    x=alt.X("AsOfDate:T", title=None, axis=alt.Axis(format="%b %Y", labelColor="#b9cee8", tickCount=3)),
                    y=alt.Y("DriftScore:Q", title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)),
                    tooltip=[alt.Tooltip("AsOfDate:T", title="Date"), alt.Tooltip("DriftScore:Q", title="Drift", format=".3f")],
                )
            )
            rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[6, 4], color="#d99a2f").encode(y="y:Q")
            st.altair_chart((line + rule).properties(height=210), use_container_width=True)
            st.caption("Higher values mean inputs are shifting further from normal ranges, so scores are less reliable.")
    else:
        st.caption("Drift trend unavailable.")

    missing_fcf = None
    try:
        with open(MODEL_HEALTH_PATH, "r", encoding="utf-8") as f:
            mh = json.load(f)
        missing_fcf = (
            mh.get("feature_availability_indicators", {})
            .get("missing_counts", {})
            .get("FreeCashFlow_Margin")
        )
    except Exception:
        missing_fcf = None

    if isinstance(missing_fcf, (int, float)) and missing_fcf > 0:
        st.markdown("### Data Quality Note")
        st.markdown(
            "⚠️ **Some data is missing.** Free cash flow data is currently unavailable for part of the universe. "
            "Scores that rely on this metric may be slightly less precise, but this does not invalidate the full signal."
        )


_show_signal_banner()

stock_tab, fi_tab, mi_tab = st.tabs(
    [
        "Stock Screener",
        "Bond & Treasury Screener",
        "Decision Intelligence",
    ]
)
with stock_tab:
    _show_stock_tab()
with fi_tab:
    _show_fixed_income_tab()
with mi_tab:
    _show_market_intelligence_tab()



