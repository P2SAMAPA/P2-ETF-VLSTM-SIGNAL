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
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ETF VLSTM Signal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

  :root {
    --bg:        #0a0e14;
    --bg2:       #111720;
    --bg3:       #1a2332;
    --border:    #1e2d42;
    --accent:    #00d4ff;
    --accent2:   #7b61ff;
    --green:     #00e5a0;
    --red:       #ff4d6d;
    --amber:     #ffc857;
    --text:      #e8edf5;
    --muted:     #6b7e96;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 1.5rem;
  }
  .stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  /* Expanders */
  .streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.05em !important;
    color: var(--muted) !important;
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
  }
  .streamlit-expanderContent {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    padding: 1rem !important;
  }

  /* Metrics */
  [data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  [data-testid="metric-container"] label {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.3rem !important;
    color: var(--text) !important;
  }

  /* Dataframes */
  .stDataFrame { border: 1px solid var(--border) !important; border-radius: 6px; }

  /* Cards */
  .card    { background:var(--bg2); border:1px solid var(--border); border-radius:8px; padding:1.25rem 1.5rem; margin-bottom:0.75rem; }
  .card-sm { background:var(--bg3); border:1px solid var(--border); border-radius:6px; padding:0.75rem 1rem; }

  /* Consensus banner */
  .banner { border-radius: 10px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; border: 1px solid; }
  .banner-strong   { background: #001a2e; border-color: #00d4ff44; }
  .banner-majority { background: #0f1a10; border-color: #00e5a044; }
  .banner-split    { background: #1a1200; border-color: #ffc85744; }
  .banner-none     { background: var(--bg2); border-color: var(--border); }

  .banner-label  { font-family:var(--mono); font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:0.4rem; }
  .banner-signal { font-family:var(--mono); font-size:2.6rem; font-weight:700; letter-spacing:0.04em; line-height:1; }
  .banner-meta   { font-family:var(--sans); font-size:0.85rem; color:var(--muted); margin-top:0.5rem; }

  .sig-strong   { color: #00d4ff; }
  .sig-majority { color: #00e5a0; }
  .sig-split    { color: #ffc857; }
  .sig-none     { color: var(--muted); }

  /* Pulse dot */
  .dot-live { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); margin-right:6px; animation:pulse 2s infinite; vertical-align:middle; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

  /* Section labels */
  .sec { font-family:var(--mono); font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); padding-bottom:0.35rem; margin:1.2rem 0 0.8rem; }

  h1,h2,h3 { font-family:var(--mono) !important; }
  h1 { font-size:1.1rem !important; letter-spacing:0.08em !important; color:var(--text) !important; }
  h2 { font-size:0.82rem !important; letter-spacing:0.1em !important; color:var(--muted) !important; text-transform:uppercase !important; }
  h3 { font-size:0.72rem !important; letter-spacing:0.08em !important; color:var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

ETF_COLORS = {
    "TLT": "#00d4ff",
    "VNQ": "#7b61ff",
    "SLV": "#c0c0c0",
    "GLD": "#ffc857",
    "HYG": "#ff4d6d",
    "LQD": "#00e5a0",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#6b7e96", size=10),
    margin=dict(l=6, r=6, t=28, b=6),
    xaxis=dict(gridcolor="#1e2d42", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#1e2d42", showline=False, zeroline=False),
)


def etf_color(etf: str) -> str:
    return ETF_COLORS.get(etf, "#6b7e96")

def fmt_pct(v, decimals=1):
    if v is None: return "—"
    return f"{v*100:+.{decimals}f}%"

def fmt_f(v, decimals=2):
    if v is None: return "—"
    return f"{v:.{decimals}f}"


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Chart helpers ─────────────────────────────────────────────────────────────

def proba_bar_chart(proba: dict, height=155) -> go.Figure:
    etfs   = list(proba.keys())
    vals   = [proba[e] for e in etfs]
    colors = [etf_color(e) for e in etfs]
    fig = go.Figure(go.Bar(
        x=etfs, y=vals,
        marker_color=colors, marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(family="Space Mono", size=9, color="#6b7e96"),
    ))
    fig.update_layout(**PLOTLY_BASE, height=height, showlegend=False,
                      yaxis=dict(visible=False, range=[0, max(vals)*1.4]),
                      margin=dict(l=4, r=4, t=6, b=4))
    return fig


def vsn_bar_chart(vsn_top: list, height=None) -> go.Figure | None:
    if not vsn_top:
        return None
    vsn_top = vsn_top[:10]
    names   = [f["feature"] for f in reversed(vsn_top)]
    weights = [f["weight"]  for f in reversed(vsn_top)]
    colors  = []
    for n in names:
        if "macd" in n:   colors.append("#7b61ff")
        elif "_r" in n:   colors.append("#00d4ff")
        elif "_vol" in n: colors.append("#ffc857")
        else:             colors.append("#00e5a0")
    fig = go.Figure(go.Bar(
        x=weights, y=names, orientation="h",
        marker_color=colors, marker_line_width=0,
    ))
    h = height or max(120, len(names) * 22)
    fig.update_layout(**PLOTLY_BASE, height=h, showlegend=False,
                      xaxis=dict(visible=False),
                      yaxis=dict(tickfont=dict(family="Space Mono", size=9, color="#6b7e96")),
                      margin=dict(l=4, r=4, t=6, b=4))
    return fig


def dist_bar_chart(live_sigs: list, height=190) -> go.Figure:
    cnt = Counter(live_sigs)
    etfs = sorted(cnt.keys(), key=lambda e: -cnt[e])
    vals = [cnt[e] for e in etfs]
    colors = [etf_color(e) for e in etfs]
    fig = go.Figure(go.Bar(
        x=etfs, y=vals,
        marker_color=colors, marker_line_width=0,
        text=vals, textposition="outside",
        textfont=dict(family="Space Mono", size=10, color="#6b7e96"),
    ))
    fig.update_layout(**PLOTLY_BASE, height=height, showlegend=False,
                      title="Live Signal Distribution Across Windows",
                      title_font=dict(family="Space Mono", size=10, color="#6b7e96"),
                      yaxis=dict(visible=False, range=[0, max(vals)*1.3]),
                      margin=dict(l=6, r=6, t=34, b=6))
    return fig


# ── Consensus banner ──────────────────────────────────────────────────────────

def render_banner(consensus: dict, stream_name: str):
    if not consensus or not consensus.get("signal"):
        st.markdown(f"""<div class="banner banner-none">
          <div class="banner-label">{stream_name}</div>
          <div class="banner-signal sig-none">—</div>
          <div class="banner-meta">No data available</div>
        </div>""", unsafe_allow_html=True)
        return

    sig      = consensus.get("signal", "—")
    strength = consensus.get("strength", "")
    score    = consensus.get("score_pct", 0)
    total    = consensus.get("total_windows", 0)
    votes    = consensus.get("top_votes", 0)
    vote_counts = consensus.get("vote_counts", {})
    today    = datetime.now().strftime("%Y-%m-%d")

    if "Strong"   in strength: b_cls, s_cls = "banner-strong",   "sig-strong"
    elif "Majority" in strength: b_cls, s_cls = "banner-majority", "sig-majority"
    else:                        b_cls, s_cls = "banner-split",    "sig-split"

    st.markdown(f"""<div class="banner {b_cls}">
      <div class="banner-label"><span class="dot-live"></span>
        {stream_name} · live consensus · {today}
      </div>
      <div class="banner-signal {s_cls}">{sig}</div>
      <div class="banner-meta">
        {strength} &nbsp;·&nbsp; {votes}/{total} windows agree &nbsp;·&nbsp; {score:.0f}% agreement
      </div>
    </div>""", unsafe_allow_html=True)

    # Vote breakdown chips
    cols = st.columns(len(vote_counts) or 1)
    for i, (etf, count) in enumerate(sorted(vote_counts.items(), key=lambda x: -x[1])):
        pct = count / total * 100 if total else 0
        c = etf_color(etf)
        cols[i].markdown(f"""<div class="card-sm" style="border-color:{c}33;">
          <div style="font-family:var(--mono);font-size:1.05rem;color:{c};font-weight:700;">{etf}</div>
          <div style="font-size:0.72rem;color:var(--muted);">{count} votes &nbsp; {pct:.0f}%</div>
          <div style="margin-top:5px;height:3px;background:#1a2332;border-radius:2px;">
            <div style="width:{pct:.0f}%;height:3px;background:{c};border-radius:2px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)


# ── Per-window expander ───────────────────────────────────────────────────────

def render_window(r: dict, idx: int):
    label      = r.get("label", f"Window {idx+1}")
    live       = r.get("live_signal") or {}
    sig        = live.get("signal", "—")
    conf       = live.get("confidence", 0)
    proba      = live.get("proba", {})
    vsn_top    = live.get("vsn_top_features", [])
    ann_ret    = r.get("ann_return")
    sharpe     = r.get("sharpe")
    max_dd     = r.get("max_dd")
    hit_rate   = r.get("hit_rate")
    n_days     = r.get("n_days", 0)
    val_sharpe = r.get("val_sharpe", 0)
    loss_mode  = r.get("loss_mode", "?")
    option     = r.get("option", "?")
    lookback   = r.get("lookback", "?")
    epochs_run = r.get("epochs_run", "?")
    all_models = r.get("all_models", [])
    audit      = r.get("audit_trail", [])
    sig_color  = etf_color(sig)

    title = (
        f"{label}  ·  Live → {sig}  ({conf*100:.0f}%)  "
        f"·  Ann {fmt_pct(ann_ret)}  ·  Sharpe {fmt_f(sharpe)}  ·  DD {fmt_pct(max_dd)}"
    )

    with st.expander(title, expanded=(idx == 0)):
        left, right = st.columns([3, 2])

        with left:
            st.markdown('<div class="sec">Live Signal — next trading day</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""
              <div style="font-family:var(--mono);font-size:2.2rem;font-weight:700;
                          color:{sig_color};line-height:1.1;">{sig}</div>
              <div style="font-size:0.65rem;color:var(--muted);font-family:var(--mono);
                          letter-spacing:0.1em;margin-top:3px;">TOP SIGNAL</div>
            """, unsafe_allow_html=True)
            c2.metric("Confidence",  f"{conf*100:.1f}%")
            c3.metric("Loss Mode",   loss_mode.upper())
            c4.metric("Option",      f"Opt {option}")

            if proba:
                st.plotly_chart(proba_bar_chart(proba), use_container_width=True,
                                config={"displayModeBar": False})

            st.markdown('<div class="sec">Backtest — test set performance</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ann. Return", fmt_pct(ann_ret))
            m2.metric("Sharpe",      fmt_f(sharpe))
            m3.metric("Max DD",      fmt_pct(max_dd))
            m4.metric("Hit Rate",    f"{hit_rate*100:.1f}%" if hit_rate is not None else "—")

            st.markdown(f"""<div style="font-size:0.68rem;color:var(--muted);
                            font-family:var(--mono);margin-top:0.5rem;">
              {n_days} test days &nbsp;·&nbsp; lookback {lookback}d &nbsp;·&nbsp;
              val_sharpe {val_sharpe:.3f} &nbsp;·&nbsp; {epochs_run} epochs
            </div>""", unsafe_allow_html=True)

        with right:
            if vsn_top:
                st.markdown('<div class="sec">VSN Attention — top features</div>', unsafe_allow_html=True)
                fig = vsn_bar_chart(vsn_top)
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                st.markdown("""<div style="font-size:0.62rem;color:var(--muted);
                                font-family:var(--mono);display:flex;gap:12px;flex-wrap:wrap;">
                  <span style="color:#00d4ff;">■ Returns</span>
                  <span style="color:#7b61ff;">■ MACD</span>
                  <span style="color:#ffc857;">■ Vol</span>
                  <span style="color:#00e5a0;">■ Macro/Other</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="sec">VSN Attention</div>', unsafe_allow_html=True)
                st.caption("Not available for this window")

        # All 4 models comparison
        if all_models and len(all_models) > 1:
            st.markdown('<div class="sec">All 4 Models — this window</div>', unsafe_allow_html=True)
            rows = []
            for m in all_models:
                ml = m.get("live_signal") or {}
                rows.append({
                    "Option":      m.get("option", "?"),
                    "Loss":        (m.get("loss_mode") or "?").upper(),
                    "Live Signal": ml.get("signal", "—"),
                    "Conf %":      f"{ml.get('confidence', 0)*100:.1f}",
                    "Val Sharpe":  f"{m.get('val_sharpe', 0):.3f}",
                    "Ann Ret":     fmt_pct(m.get("ann_return")),
                    "Sharpe":      fmt_f(m.get("sharpe")),
                    "Max DD":      fmt_pct(m.get("max_dd")),
                    "Winner":      "★" if m.get("val_sharpe") == r.get("val_sharpe") else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                         column_config={"Winner": st.column_config.TextColumn(width="small")})

        # Audit trail
        if audit:
            st.markdown('<div class="sec">Recent predictions — last 30 test-set days</div>',
                        unsafe_allow_html=True)
            df_a = pd.DataFrame(audit)
            if not df_a.empty:
                if "proba" in df_a.columns:
                    df_a = df_a.drop(columns=["proba"])
                df_a["correct"]    = df_a["correct"].map({True: "✓", False: "✗"})
                df_a["return_pct"] = df_a["return_pct"].apply(
                    lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) else x
                )
                st.dataframe(df_a, use_container_width=True, hide_index=True, height=220)


# ── Stream tab ────────────────────────────────────────────────────────────────

def render_stream(stream_data: dict, stream_name: str):
    if not stream_data:
        st.info("No data. Ensure HF_TOKEN is set and a training run has completed.")
        return

    consensus = stream_data.get("consensus", {})
    windows   = stream_data.get("windows", [])

    render_banner(consensus, stream_name)

    if not windows:
        st.info("No window results available.")
        return

    # Summary row
    valid_ret  = [w.get("ann_return", 0) for w in windows if w.get("ann_return") is not None]
    valid_sh   = [w.get("sharpe",     0) for w in windows if w.get("sharpe")     is not None]
    live_sigs  = [
        w.get("live_signal", {}).get("signal")
        for w in windows if w.get("live_signal")
    ]
    live_sigs_clean = [s for s in live_sigs if s]

    top_etf, top_count = ("—", 0)
    if live_sigs_clean:
        top_etf, top_count = Counter(live_sigs_clean).most_common(1)[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Windows",          len(windows))
    c2.metric("Avg Ann Return",   f"{np.mean(valid_ret)*100:.1f}%" if valid_ret else "—")
    c3.metric("Avg Sharpe",       f"{np.mean(valid_sh):.2f}"       if valid_sh  else "—")
    c4.metric("Top Live Signal",  top_etf)
    c5.metric("Agreement",        f"{top_count}/{len(windows)}")

    st.markdown("<br>", unsafe_allow_html=True)

    if live_sigs_clean:
        col_chart, col_space = st.columns([2, 3])
        with col_chart:
            st.plotly_chart(dist_bar_chart(live_sigs_clean), use_container_width=True,
                            config={"displayModeBar": False})

    # Per-window breakdown
    st.markdown('<div class="sec">Per-Window Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size:0.78rem;color:var(--muted);'>"
        f"{len(windows)} windows — each trained on its own date range, "
        f"each predicts tomorrow independently</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    for i, w in enumerate(windows):
        render_window(w, i)


# ── History tab ───────────────────────────────────────────────────────────────

def render_history(hist_df: pd.DataFrame):
    if hist_df.empty:
        st.info("No history yet — history builds after the first training run.")
        return

    st.markdown("## Signal History")

    col_e, col_s = st.columns(2)

    for col, prefix, label in [
        (col_e, "exp", "Expanding Stream"),
        (col_s, "shr", "Shrinking Stream"),
    ]:
        with col:
            st.markdown(f"### {label}")
            sig_col = f"{prefix}_signal"
            if sig_col in hist_df.columns:
                fig = go.Figure()
                for etf in hist_df[sig_col].dropna().unique():
                    mask = hist_df[sig_col] == etf
                    fig.add_trace(go.Scatter(
                        x=hist_df[mask]["run_date"],
                        y=[etf] * mask.sum(),
                        mode="markers",
                        marker=dict(color=etf_color(etf), size=12, symbol="circle"),
                        name=etf,
                    ))
                fig.update_layout(**PLOTLY_BASE, height=200, showlegend=True,
                                  title=f"{label} — daily consensus",
                                  title_font=dict(family="Space Mono", size=10, color="#6b7e96"),
                                  legend=dict(font=dict(family="Space Mono", size=9)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Full History Table</div>', unsafe_allow_html=True)
    st.dataframe(
        hist_df.sort_values("run_date", ascending=False),
        use_container_width=True, hide_index=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:1.5rem;">
      <div style="font-family:var(--mono);font-size:1.25rem;font-weight:700;
                  letter-spacing:0.06em;color:var(--text);">
        P2 · ETF · VLSTM · SIGNAL
      </div>
      <div style="font-family:var(--mono);font-size:0.62rem;letter-spacing:0.12em;
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
    status_color = "#00e5a0" if status_ok else "#ff4d6d"
    status_msg   = (f"Last run: {run_date} &nbsp;·&nbsp; Data through: {data_through}"
                    if status_ok else "No data — check HF_TOKEN environment variable")

    st.markdown(f"""
    <div style="font-family:var(--mono);font-size:0.66rem;color:{status_color};
                margin-bottom:1.5rem;letter-spacing:0.08em;">
      <span class="dot-live" style="background:{status_color};"></span>{status_msg}
    </div>""", unsafe_allow_html=True)

    tab_exp, tab_shr, tab_hist = st.tabs([
        "📈  EXPANDING STREAM",
        "📉  SHRINKING STREAM",
        "🕐  HISTORY",
    ])

    with tab_exp:
        render_stream(exp_data, "Expanding Stream")

    with tab_shr:
        render_stream(shr_data, "Shrinking Stream")

    with tab_hist:
        render_history(hist_df)


if __name__ == "__main__":
    main()
