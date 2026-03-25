"""
app.py
P2-ETF-VLSTM-SIGNAL

Streamlit display-only UI. No training happens here.
Reads latest.json + history.parquet from HuggingFace output dataset.

Layout:
  Tab 1 — Expanding Stream  (2008→2010 … 2008→2025)
  Tab 2 — Shrinking Stream  (2008→2025 … 2024→2025)
  Tab 3 — History

Each stream tab:
  - Headline consensus banner
  - Signal distribution chart
  - Per-window scrollable breakdown (each window expandable)
    showing: live signal, probabilities, backtest metrics,
             VSN top features, all 4 model comparison, audit trail
"""

import os
import json
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas_market_calendars as mcal  # for proper trading days


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ETF VLSTM Signal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme & CSS (unchanged) ───────────────────────────────────────────────────
# ... (keep the existing CSS block, unchanged) ...


# ── Constants ─────────────────────────────────────────────────────────────────

ETF_COLORS = {
    "TLT": "#0077cc",
    "VNQ": "#5b3fd4",
    "SLV": "#7a8a9e",
    "GLD": "#c97d00",
    "HYG": "#d6294a",
    "LQD": "#00965a",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#5a6a80", size=10),
)

_AXIS_STYLE = dict(gridcolor="#d1d9e6", showline=False, zeroline=False)


def etf_color(etf: str) -> str:
    return ETF_COLORS.get(etf, "#5a6a80")

def fmt_pct(v, decimals=1):
    if v is None: return "—"
    return f"{v*100:+.{decimals}f}%"

def fmt_f(v, decimals=2):
    if v is None: return "—"
    return f"{v:.{decimals}f}"


# ── Next trading day helper (using NYSE calendar) ─────────────────────────────

def next_trading_day(from_date: pd.Timestamp) -> pd.Timestamp:
    """
    Return the next NYSE trading day after the given date.
    Uses pandas_market_calendars for holiday awareness.
    """
    nyse = mcal.get_calendar("NYSE")
    # Get schedule for the next 10 days
    schedule = nyse.schedule(start_date=from_date, end_date=from_date + timedelta(days=10))
    trading_days = schedule.index
    # Find the first trading day strictly after from_date
    next_days = trading_days[trading_days > from_date]
    if len(next_days) > 0:
        return next_days[0]
    # Fallback: skip weekends only (should not happen with 10-day window)
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── Data loading (unchanged) ─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data() -> dict:
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        return {}
    try:
        from writer import load_latest
        return load_latest(hf_token)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return {}


@st.cache_data(ttl=300)
def load_history_df() -> pd.DataFrame:
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        return pd.DataFrame()
    try:
        from writer import load_history
        return load_history(hf_token)
    except Exception:
        return pd.DataFrame()


# ── Chart helpers (unchanged) ─────────────────────────────────────────────────
# ... (keep all existing chart functions: proba_bar_chart, vsn_bar_chart, dist_bar_chart, etc.) ...


# ── Consensus banner (modified to use data_through date) ─────────────────────

def render_banner(consensus: dict, stream_name: str, data_through: str):
    if not consensus or not consensus.get("signal"):
        st.markdown(f"""<div class="banner banner-none">
          <div class="banner-label">{stream_name}</div>
          <div class="banner-signal sig-none">—</div>
          <div class="banner-meta">No data available</div>
        </div>""", unsafe_allow_html=True)
        return

    # Compute the next trading day from the last data date
    if data_through and data_through != "—":
        try:
            last_date = pd.Timestamp(data_through)
            next_date = next_trading_day(last_date).strftime("%Y-%m-%d")
        except Exception:
            next_date = data_through  # fallback
    else:
        # Fallback to today's next trading day (original logic)
        from datetime import timezone
        est_now = datetime.now(timezone.utc) - timedelta(hours=5)
        today = est_now.date()
        days_ahead = {4: 3, 5: 2, 6: 1}.get(today.weekday(), 1)
        next_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    sig      = consensus.get("signal", "—")
    strength = consensus.get("strength", "")
    score    = consensus.get("score_pct", 0)
    total    = consensus.get("total_windows", 0)
    votes    = consensus.get("top_votes", 0)
    vote_counts = consensus.get("vote_counts", {})

    if "Strong"   in strength: b_cls, s_cls = "banner-strong",   "sig-strong"
    elif "Majority" in strength: b_cls, s_cls = "banner-majority", "sig-majority"
    else:                        b_cls, s_cls = "banner-split",    "sig-split"

    st.markdown(f"""<div class="banner {b_cls}">
      <div class="banner-label"><span class="dot-live"></span>
        {stream_name} · live consensus · {next_date}
      </div>
      <div class="banner-signal {s_cls}">{sig}</div>
      <div class="banner-meta">
        {strength} &nbsp;·&nbsp; {votes}/{total} windows agree &nbsp;·&nbsp; {score:.0f}% agreement
      </div>
    </div>""", unsafe_allow_html=True)

    # Vote breakdown chips (unchanged)
    cols = st.columns(len(vote_counts) or 1)
    for i, (etf, count) in enumerate(sorted(vote_counts.items(), key=lambda x: -x[1])):
        pct = count / total * 100 if total else 0
        c = etf_color(etf)
        cols[i].markdown(f"""<div class="card-sm" style="border-color:{c}55;">
          <div style="font-family:var(--mono);font-size:1.3rem;color:{c};font-weight:700;">{etf}</div>
          <div style="font-size:0.90rem;color:var(--muted);">{count} votes &nbsp; {pct:.0f}%</div>
          <div style="margin-top:5px;height:3px;background:#d1d9e6;border-radius:2px;">
            <div style="width:{pct:.0f}%;height:3px;background:{c};border-radius:2px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)


# ── Per-window expander (unchanged) ───────────────────────────────────────────
# ... (keep all existing render_window code) ...


# ── Stream tab (modified to pass data_through to banner) ─────────────────────

def render_stream(stream_data: dict, stream_name: str, data_through: str):
    if not stream_data:
        st.info("No data. Ensure HF_TOKEN is set and a training run has completed.")
        return

    consensus = stream_data.get("consensus", {})
    windows   = stream_data.get("windows", [])

    render_banner(consensus, stream_name, data_through)

    if not windows:
        st.info("No window results available.")
        return

    # ... (rest of the function unchanged) ...


# ── History tab (unchanged) ───────────────────────────────────────────────────
# ... (keep render_history as is) ...


# ── Main (unchanged except passing data_through) ──────────────────────────────

def main():
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:1.5rem;">
      <div style="font-family:var(--mono);font-size:1.25rem;font-weight:700;
                  letter-spacing:0.06em;color:var(--text);">
        P2 · ETF · VLSTM · SIGNAL
      </div>
      <div style="font-family:var(--mono);font-size:0.80rem;letter-spacing:0.12em;
                  color:var(--muted);text-transform:uppercase;">
        VSN + LSTM &nbsp;·&nbsp; Expanding &amp; Shrinking Windows &nbsp;·&nbsp; Display Only
      </div>
    </div>
    """, unsafe_allow_html=True)

    data    = load_data()
    hist_df = load_history_df()

    run_date     = data.get("run_date",     "—") if data else "—"
    data_through = data.get("data_through", "—") if data else "—"
    exp_data     = data.get("expanding",    {})  if data else {}
    shr_data     = data.get("shrinking",    {})  if data else {}

    status_ok    = bool(data)
    status_color = "#00965a" if status_ok else "#d6294a"
    status_msg   = (f"Last run: {run_date} &nbsp;·&nbsp; Data through: {data_through}"
                    if status_ok else "No data — check HF_TOKEN environment variable")

    st.markdown(f"""
    <div style="font-family:var(--mono);font-size:0.82rem;color:{status_color};
                margin-bottom:1.5rem;letter-spacing:0.08em;">
      <span class="dot-live" style="background:{status_color};"></span>{status_msg}
    </div>""", unsafe_allow_html=True)

    tab_exp, tab_shr, tab_hist = st.tabs([
        "📈  EXPANDING STREAM",
        "📉  SHRINKING STREAM",
        "🕐  HISTORY",
    ])

    with tab_exp:
        render_stream(exp_data, "Expanding Stream", data_through)

    with tab_shr:
        render_stream(shr_data, "Shrinking Stream", data_through)

    with tab_hist:
        render_history(hist_df)


if __name__ == "__main__":
    main()
