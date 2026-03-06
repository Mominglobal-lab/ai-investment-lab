## User Manual: Tab 1 - Stock Screener

### 1. Overview
The **Stock Screener** tab helps you filter and rank stocks using fundamental metrics and model outputs (Quality Score/Tier).  
It combines cached market/fundamental data with optional model artifacts, then applies your filters to return matching securities.

### 2. Purpose
Use this tab to:
- Narrow a stock universe (S&P 500 or Nasdaq 100)
- Screen by quality, growth, valuation, profitability, and size
- Sort results by market cap or model quality
- Inspect ticker-level details quickly

### 3. What This Tab Calculates
The tab performs these calculation steps:

1. Loads stock data from cache (or refreshes if needed).
2. Standardizes schema and computes derived metrics.
3. Optionally merges model outputs (`QualityScore`, `QualityTier`).
4. Applies your filter conditions.
5. Sorts matches and displays a formatted results table.

It is primarily a **rule-based filtering/ranking engine** (not a forecasting engine).

### 4. Key Derived Formulas Used

1. **PEG Ratio**
\[
\text{PEG\_Ratio} = \frac{\text{PE\_Ratio}}{\text{Earnings\_Growth\_Pct}}
\]
Applied only when `Earnings_Growth_Pct > 0`, otherwise left empty.

2. **Rule of 40**
\[
\text{Rule\_of\_40} = \text{Revenue\_Growth\_YoY\_Pct} + (\text{EBITDA\_Margin} \times 100)
\]
`EBITDA_Margin` is stored as decimal and converted to percent points in this formula.

3. **Display Conversions**
- `MarketCap (B) = MarketCap / 1e9`
- `EBITDA Margin (Pct) = EBITDA_Margin * 100`
- `ROE (Pct) = ROE * 100`

### 5. User Selections and Meaning

#### Top Controls
- **Universe**: Chooses stock set (`S&P 500` or `Nasdaq 100`).
- **Enrich metadata (slower)**: Adds extra metadata during refresh, increases runtime.
- **Refresh Stock Data**: Forces data refresh now.
- **Reset Filters**: Returns all filter values to defaults.

#### Main Filters
- **Search ticker/company**: Text match on ticker or company name.
- **Sectors**: Include only selected sectors.
- **Rule of 40 min**: Minimum acceptable Rule of 40.

#### Advanced Screening Filters
- **EBITDA margin min**: Minimum EBITDA margin (decimal scale in filter input logic).
- **ROE min**: Minimum return on equity (decimal scale).
- **Revenue growth YoY min (Pct)**: Minimum revenue growth in percent points.
- **Earnings growth min (Pct)**: Minimum earnings growth in percent points.
- **P/E ratio min**: Minimum P/E.
- **PEG ratio min**: Minimum PEG.
- **Market cap min (B)**: Minimum market cap in billions.
- **Market cap max (B)**: Maximum market cap in billions.
- **Require complete data**:
  - `Off`: allows missing metric values to remain in results if they don’t violate active filters.
  - `On`: drops rows missing critical screening fields.
- **Quality score min**: Minimum model quality score.
- **Sort by**: `MarketCap` or `QualityScore` (descending).

### 6. How Filtering Logic Works (Important)
For each active minimum/maximum filter:
- If **Require complete data = Off**, rows with missing values may still be kept.
- If **Require complete data = On**, missing critical values are removed and strict filtering is enforced.

Then results are sorted descending by your selected sort key and shown as **Matches: N**.

### 7. Output Table Columns (Typical)
- Identity: `Ticker`, `Company`, `Sector`
- Model: `QualityScore`, `QualityTier`
- Price/Size: `Close`, `MarketCap (B)`
- Growth: `Revenue_Growth_YoY_Pct`, `Earnings_Growth_Pct`
- Valuation: `PE_Ratio`, `PEG_Ratio`
- Combined rule: `Rule_of_40`
- Profitability: `EBITDA Margin (Pct)`, `ROE (Pct)`

### 8. Ticker Details Section
Selecting a ticker in **Ticker details** shows quick per-ticker info:
- Company, Sector
- Close
- P/E
- PEG
- Rule of 40

---

## BEGIN TAB 2 - Bond & Treasury Screener User Manual

### 1. Overview
The **Bond & Treasury Screener** tab filters fixed-income instruments (Treasuries and bond ETFs) using yield, duration, maturity bucket, and search criteria.

### 2. Purpose
Use this tab to:
- Find fixed-income instruments that match yield and duration constraints
- Narrow instruments by maturity profile
- Compare ETF/bond characteristics in one sortable table
- Inspect single-instrument details and duration shock sensitivity

### 3. What This Tab Calculates
This tab performs:
1. Load fixed-income cache (or refresh if stale/missing/manual refresh requested).
2. Convert numeric fields (`Price`, `Yield_Pct`, `Duration_Years`, `Expense_Ratio_Pct`, `AUM`) to numeric types.
3. Apply user filters (search, maturity bucket, minimum yield, maximum duration).
4. Sort by `Yield_Pct` descending.
5. Compute display field `AUM (B)`.
6. Display matching instruments and optional instrument detail panel.

This is a **screening and ranking tab** (not a price forecast model).

### 4. Formula(s) Used
1. **AUM in Billions (display)**
\[
\text{AUM (B)} = \frac{\text{AUM}}{1,000,000,000}
\]

2. **Estimated price impact for +100 bps move** (detail panel)
\[
\text{Estimated Impact (\%)} = -\text{Duration\_Years}
\]
Interpretation: a +1.00% rate move implies an approximate price decline equal to duration in percent terms.

### 5. User Selections and Meaning

#### Top Controls
- **Universe**: Chooses the fixed-income source set (`US Treasuries` or `Bond ETFs`).
- **Refresh Fixed-Income Data**: Forces data refresh from pipeline now.

#### Filters
- **Search symbol/name**: Text filter against `Symbol` or `Name`.
- **Maturity bucket**: Multi-select filter using categories such as `0-1Y`, `1-3Y`, `3-7Y`, `7-10Y`, `20Y+`.
- **Yield min (%)**: Minimum `Yield_Pct` threshold.
- **Duration max (yrs)**: Maximum `Duration_Years` threshold.

### 6. Filter Logic
Filtering order:
1. Search text match on symbol or name.
2. Maturity bucket inclusion (if selected).
3. Keep rows where `Yield_Pct >= Yield min`.
4. Keep rows where `Duration_Years <= Duration max`.
5. Sort remaining rows by `Yield_Pct` descending.

### 7. Output Table Columns
Typical displayed columns:
- `Symbol`
- `Name`
- `Universe`
- `Type`
- `Price`
- `Yield_Pct`
- `Duration_Years`
- `Maturity_Bucket`
- `Expense_Ratio_Pct`
- `AUM (B)`

### 8. Instrument Details Panel
Selecting an instrument in **Instrument details** shows:
- Name and symbol
- Yield (%)
- Duration (years)
- Expense ratio (%)
- AUM
- Estimated +100 bps rate shock impact using duration approximation

## END TAB 2 - Bond & Treasury Screener User Manual

---

## BEGIN TAB 3 - Portfolio Decision Simulator User Manual

### 1. Overview
The **Portfolio Decision Simulator** tab tests how a custom portfolio would have behaved over a selected historical window, with optional Monte Carlo scenario analysis.

### 2. Purpose
Use this tab to:
- Build a portfolio from selected tickers and weights
- Compare performance versus a benchmark (for example, `SPY`)
- Evaluate risk and drawdown behavior
- Run probabilistic scenarios (Monte Carlo mode)
- Export a decision brief artifact

### 3. What This Tab Calculates
The tab performs:
1. Reads aligned price history from `prices_cache`.
2. Builds portfolio weights from selected weighting method.
3. Computes daily portfolio returns and value path.
4. Applies rebalance logic (`None` or `Monthly`).
5. Computes drawdown series and summary risk/performance metrics.
6. Optionally computes Monte Carlo outcome distributions.
7. Produces decision insights and exportable brief output.

### 4. User Selections and Meaning

#### Action Buttons
- **Refresh Data**: Rebuilds prices cache with latest available data.
- **Run Simulation**: Executes the simulation with current settings.
- **Export Decision Brief**: Writes HTML/JSON brief to `data/run_artifacts`.

#### Portfolio Inputs
- **Tickers (comma-separated)**: Portfolio members, e.g., `AAPL, MSFT, NVDA`.
- **Weighting**:
  - `Equal weight`: each ticker gets equal share.
  - `Market cap weight`: uses market-cap proportional weights from fundamentals cache.
  - `Manual weights`: user-entered weights matching ticker order.
- **Manual weights (when enabled)**: comma-separated weights, count must match ticker count.
- **Benchmark**: Ticker used for comparison path and correlation/beta metrics.

#### Simulation Configuration
- **Lookback period**: Historical window in years (`1, 3, 5, 10`).
- **Rebalance**:
  - `None`: no periodic rebalance after initial allocation.
  - `Monthly`: reset back to target weights at each month boundary.
- **Simulation mode**:
  - `Historical`: backtest over observed history only.
  - `Monte Carlo`: adds randomized path scenarios.
- **Starting capital ($)**: Initial portfolio value.
- **Strict missing-data mode**:
  - `Off`: drops unavailable tickers and renormalizes remaining weights.
  - `On`: fails run if any selected ticker is missing.

#### Monte Carlo Inputs (visible in Monte Carlo mode)
- **Number of simulations**: number of paths to sample.
- **Horizon years**: forward horizon converted to trading days.

### 5. Core Formulas Used

1. **Portfolio Daily Return**
\[
r_{p,t} = \frac{V_t}{V_{t-1}} - 1
\]
where \(V_t\) is portfolio value after applying constituent daily returns and current weights.

2. **Drawdown**
\[
\text{Drawdown}_t = \frac{V_t}{\max(V_1,\dots,V_t)} - 1
\]

3. **CAGR**
\[
\text{CAGR} = \left(\frac{V_{\text{end}}}{V_0}\right)^{1/\text{years}} - 1
\]

4. **Annualized Volatility**
\[
\sigma_{\text{ann}} = \text{Std}(r_p)\sqrt{252}
\]

5. **Sharpe Ratio**
\[
\text{Sharpe} = \frac{\text{CAGR} - r_f}{\sigma_{\text{ann}}}
\]
(`risk_free_rate` default is 0 in current implementation.)

6. **VaR 95 / CVaR 95 (daily)**
- **VaR 95** = 5th percentile of daily returns.
- **CVaR 95** = mean return in the tail where returns are below VaR 95.

7. **Benchmark Correlation / Beta**
- Correlation computed on aligned daily return series.
- Beta:
\[
\beta = \frac{\text{Cov}(r_p, r_b)}{\text{Var}(r_b)}
\]

8. **Monte Carlo Path Generation**
- Uses multivariate normal draws from historical mean vector and covariance matrix.
- Path value:
\[
V_t = V_0\prod_{i=1}^{t}(1+r_{p,i}^{MC})
\]
- Outputs percentile bands (`p05, p25, p50, p75, p95`) for ending value and max drawdown, plus probability of loss.

### 6. Output Interpretation

#### Growth Chart
- Shows portfolio value path; benchmark overlay shown when benchmark data exists.
- Optional **Risk Overlay** adds risk-score line for context.

#### Drawdown Chart
- Shows percent decline from running peak over time.

#### Performance Metrics Table
Includes:
- `CAGR`
- `volatility`
- `Sharpe_ratio`
- `max_drawdown`
- `worst_day`
- `worst_month`
- `VaR_95`
- `CVaR_95`
- `correlation_with_benchmark`
- `beta_relative_to_benchmark`

#### Monte Carlo Scenario Summary (when enabled)
- Ending value percentiles
- Max drawdown percentiles
- Probability of ending below starting capital

#### Decision Insights
Narrative checks using:
- Relative portfolio vs benchmark outcome
- Concentration proxy (HHI on weights)
- Volatility and tail-risk thresholds
- Monte Carlo probability-of-loss level

### 7. Input Validation and Edge Cases
- Negative weights are rejected.
- If selected tickers are unavailable in price cache:
  - strict mode off: dropped + remaining weights renormalized.
  - strict mode on: run fails.
- If insufficient aligned history exists after filtering, run fails with error.
- Benchmark metrics are blank if benchmark series is unavailable.

## END TAB 3 - Portfolio Decision Simulator User Manual

---

## BEGIN TAB 4 - Decision Intelligence User Manual

### 1. Overview
The **Decision Intelligence** tab is a model-artifact dashboard. It summarizes outputs from the quality, regime, and risk model pipelines and shows their latest state, trend, and health.

### 2. Purpose
Use this tab to:
- Verify model artifacts were built successfully
- Inspect quality ranking outputs
- Track regime/risk timelines
- Review recent risk signals
- Audit model registry metadata and cache health

### 3. What This Tab Calculates
This tab is mainly a **read/analyze dashboard** (it does not run portfolio simulation math).  
It performs:
1. Loads model caches:
   - `quality_scores_cache.parquet`
   - `regime_cache.parquet`
   - `risk_signals_cache.parquet`
2. Computes summary counts and latest labels for top metrics.
3. Bins quality scores into fixed ranges for distribution chart.
4. Maps regime labels into numeric codes for timeline plotting.
5. Loads and flattens `model_registry.json` and `model_health_report.json` for diagnostics.

### 4. User Selections and Meaning

#### Main Action
- **Build Model Artifacts**: Runs the decision-model pipeline and refreshes quality/regime/risk outputs.

#### No Other Manual Filters in This Tab
Most content is auto-derived from latest model artifacts.  
The tab is intended for inspection rather than parameterized screening.

### 5. Formulas / Mappings Used

1. **Quality Score Distribution Bucketing**
Quality scores are grouped into fixed bins:
- `0-20`
- `20-40`
- `40-60`
- `60-80`
- `80-100`

2. **Regime Label Numeric Mapping (for timeline y-axis)**
\[
\text{RegimeCode} =
\begin{cases}
-1 & \text{if RegimeLabel = Risk Off}\\
0 & \text{if RegimeLabel = Neutral}\\
1 & \text{if RegimeLabel = Risk On}
\end{cases}
\]

3. **Top Quality Entities**
Sorted by:
\[
\text{QualityScore} \text{ descending}
\]
Then top 30 are displayed.

4. **Recent Risk Signals**
Risk rows are sorted by `Date` ascending in preprocessing; display uses latest tail (`tail(30)`), i.e., most recent observations.

5. **Model Health Counts**
- `Fresh` = count of artifacts where status is `"fresh"`
- `Stale` = count where status is `"stale"`
- `Other` = remaining status labels

### 6. Section-by-Section Output Meaning

#### A. Header Metrics
- **Quality Rows**: number of rows loaded in quality cache.
- **Latest Regime**: most recent regime label.
- **Latest Risk**: most recent risk level.

#### B. Quality Score Distribution
- Bar chart of counts across the five score buckets.

#### C. Top Quality Entities
- Table of highest quality names (top 30 by score), including score and tier fields.

#### D. Regime Timeline
- Time-series line chart of mapped regime code (-1/0/1), colored by regime label.

#### E. Systemic Risk Timeline
- Time-series chart of `RiskScore`.
- **Recent Risk Signals** table gives recent signal rows and flags.

#### F. Model Registry
- Reads `data/model_registry.json`.
- Shows metadata such as model name/version, timestamp, training/evaluation window, and selected evaluation summary fields.

#### G. Model Health
- Reads `data/model_health_report.json`.
- Shows freshness summary counts.
- Shows cache coverage diagnostics (`exists`, `schema_ok`, row counts, and status).

### 7. Data Dependencies and Fallback Behavior
- If core caches are missing, tab warns that artifacts are not found.
- If `model_registry.json` or `model_health_report.json` is missing, that section shows an availability caption and continues.
- This tab does not attempt to infer missing model outputs from other tabs.

### 8. Interpretation Guidance
- Use **Quality Distribution + Top Entities** for candidate idea generation.
- Use **Regime + Risk timelines** to understand macro context and risk backdrop.
- Use **Model Registry + Health** before operational decisions to confirm artifact freshness and integrity.

## END TAB 4 - Decision Intelligence User Manual

---

## BEGIN TAB 5 - Explainability and Evidence User Manual

### 1. Overview
The **Explainability and Evidence** tab explains *why* model outputs were produced.  
It breaks explanations into three layers:
- Entity-level quality drivers
- Regime (operating environment) evidence
- Systemic risk evidence

### 2. Purpose
Use this tab to:
- Understand the top positive/negative factors behind a selected ticker’s quality score
- Inspect evidence points used for regime and risk conclusions
- View supporting indicator values across recent history

### 3. What This Tab Calculates
This tab is mostly interpretive and performs:
1. Loads explainability/evidence artifacts (with fallbacks to base caches if missing).
2. Parses JSON driver/evidence fields into structured tables.
3. Ranks contribution magnitudes for feature importance.
4. Maps selected evidence indicators over time for visual context.

### 4. User Selections and Meaning

#### Main Action
- **Build Explainability Artifacts**: Runs explainability pipeline (and decision models first if needed).

#### Section A: Entity Explainability
- **Ticker**: choose the entity to inspect.

#### Section B: Operating Environment Evidence
- **Regime date**: select a date to inspect regime label/confidence and evidence points.

#### Section C: Systemic Risk Evidence
- **Risk date**: select a date to inspect risk score/level and evidence points.
- **Last N days**: controls lookback window for risk trend chart.
- **Underlying indicator**: choose one evidence indicator to chart over lookback window.

### 5. Core Logic and Formulas

1. **Feature Contribution Ranking**
From `ContributionJSON`, each feature has a signed contribution value.

Intermediate metric used for ranking:
\[
\text{AbsContribution} = |\text{SignedContribution}|
\]
Features are sorted by `AbsContribution` descending; display retains signed value.

2. **Evidence Card Mapping**
Top 3 features from contribution ranking are used to fetch raw values from fundamentals data.  
If a feature ends with `_stability`, that suffix is removed before lookup.

3. **Evidence Point Parsing**
`EvidencePointsJSON` is parsed into key-value rows:
- `Indicator`
- `Value`
Formatting behavior in UI:
- Indicator names ending in `_flag` are shown as `True/False`.
- Trend/volatility style indicators (for example `BenchmarkTrend_63d`, `BenchmarkVol_63d`) are shown in percent format where applicable.

4. **Risk Indicator Time Series Build**
For selected indicator and lookback window:
- Parse each row’s `EvidencePointsJSON`
- Extract selected indicator value
- Plot value by date

### 6. Output Sections and Interpretation

#### A. Entity Explainability
Displays:
- `QualityScore`
- `QualityTier`
- `FeatureAsOfDate`
- Top positive and negative driver text
- Feature contribution table (signed contribution)
- Evidence Card (top-feature raw values with human-readable formatting; percent-style fields shown as `%`)

Interpretation:
- Positive signed contribution supports the score.
- Negative signed contribution detracts from the score.
- Large absolute values indicate stronger influence.

#### B. Operating Environment Evidence
Displays for selected date:
- `RegimeLabel`
- `ConfidenceScore` (if present)
- Trigger and short explanation text
- Indicator evidence table from JSON payload

#### C. Systemic Risk Evidence
Displays for selected date:
- `RiskScore`
- `RiskLevel`
- Top risk drivers and short explanation (if present)
- Evidence table from JSON payload
- Risk score trend over last N days
- Optional chosen indicator trend over last N days

### 7. Data Sources and Fallback Behavior
- Primary files:
  - `quality_explanations_cache.parquet`
  - `regime_evidence_cache.parquet`
  - `risk_evidence_cache.parquet`
- Fallbacks:
  - quality fallback: `quality_scores_cache.parquet`
  - regime fallback: `regime_cache.parquet`
  - risk fallback: `risk_signals_cache.parquet`

If evidence artifacts are missing, the tab still renders with available fallback data and warnings.

### 8. Practical Use Guidance
- Start with ticker-level contributions to understand stock-specific signal drivers.
- Validate macro context in Regime Evidence before acting on single-name outputs.
- Use Risk Evidence indicator trend to see whether a risk signal is persistent or temporary.

## END TAB 5 - Explainability and Evidence User Manual

---

## BEGIN TAB 6 - Uncertainty and Confidence User Manual

### 1. Overview
The **Uncertainty and Confidence** tab shows uncertainty ranges around model outputs instead of only point estimates.

It is split into:
- Quality score uncertainty
- Regime probability distribution
- Risk score uncertainty band

### 2. Purpose
Use this tab to:
- Understand confidence intervals around quality and risk outputs
- Compare regime probabilities across states (`Risk On`, `Neutral`, `Risk Off`)
- Assess stability of model labels over recent windows

### 3. What This Tab Calculates
This tab performs:
1. Loads uncertainty artifacts (with fallbacks to base caches if unavailable).
2. Selects a single row by ticker/date for display.
3. Displays quantile-style statistics (P10/P50/P90 where available).
4. Builds probability tables and charts for regime states.
5. Draws risk uncertainty band (`RiskP10` to `RiskP90`) around `RiskScore`.

### 4. User Selections and Meaning

#### Main Action
- **Build Uncertainty Artifacts**: Runs uncertainty pipeline (and decision model pipeline first if needed).

#### Section A: QualityScore Uncertainty
- **Ticker**: select entity for uncertainty view.
- **Show uncertainty band chart**: toggles the P10/P50/P90 bar chart.

#### Section B: Regime Probabilities
- **Regime probability date**: selects date for probability snapshot.

#### Section C: Risk Uncertainty
- **Risk uncertainty date**: selects date for risk uncertainty snapshot.
- **Last N days**: controls lookback horizon for risk uncertainty chart.

### 5. Core Metrics and Formulas

1. **Quantile-style metrics**
- `P10`, `P50`, `P90` represent lower, central, and upper uncertainty levels.
- Displayed for:
  - quality score (`ScoreP10`, `ScoreP50`, `ScoreP90`)
  - risk score (`RiskP10`, `RiskP50`, `RiskP90`)

2. **Regime Probability Vector**
For selected date:
- `P_RiskOn`
- `P_Neutral`
- `P_RiskOff`

These are displayed as a table and bar chart.

3. **Risk Uncertainty Band**
For lookback dates:
- Lower bound = `RiskP10`
- Upper bound = `RiskP90`
- Central line = `RiskScore`

Visual interpretation:
- Wider `RiskP90 - RiskP10` band suggests higher uncertainty.

4. **Stability Metrics**
- `TierStability`: consistency of quality tier under feature noise.
- `RegimeStability_20d`: regime stability over recent 20-day window (if present).
- `RiskLevelStability`: stability of risk level assignment (if present).

### 6. Section-by-Section Output Meaning

#### A. QualityScore Uncertainty
If uncertainty artifact is available, displays:
- `ScoreP10`, `ScoreP50`, `ScoreP90`
- `TierMostLikely`
- `TierStability`
- Optional P10/P50/P90 chart

Fallback behavior:
- If uncertainty columns are missing, shows point estimate `QualityScore` only.

#### B. Regime Probabilities
Displays:
- `RegimeLabel`
- `ConfidenceScore`
- Probability table for `Risk On`, `Neutral`, `Risk Off`
- Probability bar chart
- Optional `RegimeStability_20d`

Fallback behavior:
- If probability columns are missing, shows only label and confidence.

#### C. Risk Uncertainty
Displays:
- `RiskScore`
- `RiskP10`, `RiskP50`, `RiskP90` (if available)
- `RiskLevelMostLikely`
- `RiskLevelStability`
- Time-series uncertainty band chart over selected lookback

Fallback behavior:
- If uncertainty columns are missing, shows point `RiskScore` and `RiskLevel` only.

### 7. Data Sources and Fallbacks
- Primary uncertainty artifacts:
  - `quality_uncertainty_cache.parquet`
  - `regime_probabilities_cache.parquet`
  - `risk_uncertainty_cache.parquet`
- Fallback artifacts:
  - `quality_scores_cache.parquet`
  - `regime_cache.parquet`
  - `risk_signals_cache.parquet`

### 8. Practical Interpretation Guidance
- Prefer decisions where central estimate is strong **and** uncertainty band is tight.
- Treat high-probability regime calls with low stability carefully.
- For risk management, watch both risk level and width of uncertainty band.

## END TAB 6 - Uncertainty and Confidence User Manual

---

## BEGIN TAB 7 - Drift, Monitoring, and Early Warning User Manual

### 1. Overview
The **Drift, Monitoring, and Early Warning** tab tracks model/data health over time and highlights operational risk through alerts, drift scores, and coverage diagnostics.

### 2. Purpose
Use this tab to:
- Detect feature/signal drift early
- Review active alerts and supporting evidence
- Monitor stability of regime/risk signals
- Validate monitoring pipeline health and data coverage

### 3. What This Tab Calculates
This tab performs:
1. Loads monitoring artifacts:
   - `drift_signals_cache.parquet`
   - `alert_log.parquet`
   - optional JSON reports (`drift_report.json`, `monitoring_health_report.json`)
2. Computes summary statuses for drift severity and signal stability.
3. Filters and ranks drift rows for dashboard view.
4. Counts active alerts by severity.
5. Builds trend visualizations for selected drift metrics.
6. Computes short-window stability metrics for regime/risk series.

### 4. User Selections and Meaning

#### Main Action
- **Build Monitoring Artifacts**: runs monitoring pipeline (and model pipeline first if needed).

#### Section B: Active Alerts
- **Select alert**: choose one recent alert to inspect description/evidence/action.

#### Section C: Drift Dashboard
- **DriftLevel filter**: `All`, `Stable`, `Drift`, `Severe`
- **MetricType filter**: `All`, `Feature`, `Signal`
- **Drift metric trend**: pick one metric name to plot `DriftScore` over time.

### 5. Core Logic and Formulas

1. **Worst Drift Level**
- `Severe` if any row has `DriftLevel == Severe`
- else `Drift` if any row has `DriftLevel == Drift`
- else `Stable`

2. **Signal Stability Status**
- `Unstable` if any row where `MetricType == Signal` has `DriftLevel != Stable`
- otherwise `Stable`

3. **Active Alerts by Severity**
- `Critical`, `Warning`, `Info` counts are computed from `alert_log`.

4. **Recent Alert Window**
- Uses last 30 days of alerts by date when available.
- If none in window, falls back to latest 30 alerts.

5. **Drift Dashboard Sorting**
\[
\text{Sort by DriftScore descending}
\]
after applying selected level/type filters.

6. **Regime Flip Rate (60d)**
\[
\text{FlipRate} = \text{mean}(RegimeLabel_t \neq RegimeLabel_{t-1})
\]
computed on last 60 dated rows.

7. **RiskScore Volatility (60d)**
\[
\sigma_{60d} = \text{Std}(RiskScore)
\]
computed on last 60 dated rows (sample standard deviation).

8. **Risk Level Changes (60d)**
- Count of rows where `RiskLevel` differs from prior row over trailing 60 rows.

### 6. Section-by-Section Output Meaning

#### A. Monitoring Summary
Displays:
- **Feature Drift** (worst current drift level)
- **Signal Stability** (`Stable`/`Unstable`)
- **Active Alerts** summary (`C:x W:y I:z`)

#### B. Active Alerts
Displays:
- Recent alerts table (severity/type/title/date)
- Selected alert description
- Parsed evidence JSON table (if present)
- Suggested action text

#### C. Drift Dashboard
Displays:
- Filtered/sorted drift table
- Time-series line of `DriftScore` for selected metric

#### D. Signal Stability
Displays:
- `Regime Flip Rate (60d)`
- Average regime confidence (if available)
- `RiskScore Volatility (60d)`
- Risk-level change count
- Quality instability proxy metric (if available)

#### Monitoring Reports (optional)
If JSON reports exist, displays:
- Drift report metadata, top drifting features/signals
- Window settings and coverage summary
- Missing-count diagnostics
- Monitoring health freshness/coverage/runtime notes

### 7. Data Sources and Fallback Behavior
- Core dependency: `drift_signals_cache.parquet` (required for tab rendering)
- Optional:
  - `alert_log.parquet`
  - `drift_report.json`
  - `monitoring_health_report.json`
- If optional files are missing, related sections show availability captions and continue.

### 8. Practical Interpretation Guidance
- Prioritize investigation when **Feature Drift = Severe** or **Signal Stability = Unstable**.
- Combine alert evidence with drift trend before intervention.
- Treat repeated risk-level changes and rising 60d volatility as elevated monitoring concern.
- Use coverage/missing-count sections to distinguish model drift from data-quality issues.

## END TAB 7 - Drift, Monitoring, and Early Warning User Manual
