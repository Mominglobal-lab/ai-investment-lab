from __future__ import annotations

import numpy as np
import pandas as pd


def _same_as_previous(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    same = series.eq(prev)
    if len(same) > 0:
        same.iloc[0] = True
    return same


def build_regime_probabilities(regime_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if regime_df is None or regime_df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "RegimeLabel",
                "ConfidenceScore",
                "P_RiskOn",
                "P_Neutral",
                "P_RiskOff",
                "RegimeStability_20d",
            ]
        )

    r = regime_df.copy()
    r["Date"] = pd.to_datetime(r["Date"], errors="coerce")
    r["ConfidenceScore"] = pd.to_numeric(r["ConfidenceScore"], errors="coerce").fillna(0.55)
    r["RegimeLabel"] = r["RegimeLabel"].where(pd.notna(r["RegimeLabel"]), "Neutral")
    r["RegimeLabel"] = r["RegimeLabel"].astype(str).str.strip()
    invalid_labels = r["RegimeLabel"].str.lower().isin({"", "nan", "none"})
    r.loc[invalid_labels, "RegimeLabel"] = "Neutral"
    r = r.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    out_rows: list[dict[str, object]] = []
    same_as_previous = _same_as_previous(r["RegimeLabel"])
    stable_roll = same_as_previous.rolling(int(window), min_periods=1).mean().fillna(1.0)

    for i, row in r.iterrows():
        label = str(row["RegimeLabel"])
        conf = float(np.clip(row["ConfidenceScore"], 0.34, 0.90))
        rem = (1.0 - conf) / 2.0
        p_on, p_neu, p_off = rem, rem, rem
        if label == "Risk On":
            p_on = conf
        elif label == "Risk Off":
            p_off = conf
        else:
            p_neu = conf
        # Normalize for numeric stability.
        s = p_on + p_neu + p_off
        p_on, p_neu, p_off = p_on / s, p_neu / s, p_off / s

        out_rows.append(
            {
                "Date": row["Date"],
                "RegimeLabel": label,
                "ConfidenceScore": float(row["ConfidenceScore"]),
                "P_RiskOn": float(p_on),
                "P_Neutral": float(p_neu),
                "P_RiskOff": float(p_off),
                "RegimeStability_20d": float(stable_roll.iloc[i]),
            }
        )
    return pd.DataFrame(out_rows)

