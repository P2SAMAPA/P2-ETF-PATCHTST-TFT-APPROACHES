import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import gitlab
from io import StringIO
import engine  # your strategy module

# --- HEADER & BRANDING ---
st.set_page_config(page_title="ETF PREDICTOR: TRANSFORMER & TFT", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Transformer and TFT ETF PREDICTOR</h1>", unsafe_allow_html=True)

# --- TRADING CALENDAR ---
def get_next_trading_day():
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=datetime.now(), end_date=datetime.now() + timedelta(days=7))
        return schedule.index[0].strftime('%Y-%m-%d')
    except:
        return "Market Closed"

# --- DATA FETCHING FROM GITLAB ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        gl_token = st.secrets["GITLAB_API_TOKEN"]
        gl_id = st.secrets["GITLAB_PROJECT_ID"]
        gl = gitlab.Gitlab("https://gitlab.com", private_token=gl_token)
        project = gl.projects.get(gl_id)
        file_info = project.files.get(file_path="master_data.csv", ref="main")
        df = pd.read_csv(StringIO(file_info.decode().decode('utf-8')), index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_signals(model):
    """Load precomputed signals for the selected model from GitLab"""
    try:
        gl_token = st.secrets["GITLAB_API_TOKEN"]
        gl_id = st.secrets["GITLAB_PROJECT_ID"]
        gl = gitlab.Gitlab("https://gitlab.com", private_token=gl_token)
        project = gl.projects.get(gl_id)
        if model == "Transformer":
            file_name = "signals_transformer.csv"
        else:
            file_name = "signals_tft.csv"
        file_info = project.files.get(file_path=file_name, ref="main")
        signals = pd.read_csv(StringIO(file_info.decode().decode('utf-8')), index_col=0)
        signals.index = pd.to_datetime(signals.index)
        return signals
    except Exception as e:
        st.warning(f"Could not load signals: {e}")
        return pd.DataFrame()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Strategy Engine")
    option = st.radio("Active Model Selection", ["Option A: Transformer", "Option B: TFT"])
    model_name = "Transformer" if "Option A" in option else "TFT"

    st.subheader("Training Start Year")
    start_year = st.slider("Start Year", 2008, 2025, 2008)
    end_year = 2025  # fixed

    st.divider()
    tsl_val = st.slider("2-Day Cumulative TSL (%)", 0.0, 25.0, 10.0, step=0.5)
    txn_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05)

# --- MAIN UI ---
df = load_data()
next_open = get_next_trading_day()

if df.empty:
    st.warning("No data loaded. Please check GitLab connection and data file.")
    st.stop()

# Load precomputed signals for the selected model
signals_df = load_signals(model_name)
if signals_df.empty:
    st.warning("No signals available. Please run the signal precomputation workflow first.")
    st.stop()

# Run the strategy engine with the selected parameters
with st.spinner("Running strategy simulation..."):
    results = engine.compute_strategy_logic(df, option, (start_year, end_year), txn_cost, tsl_val, signals_df)

if results is None:
    st.error("No data available for the selected period.")
    st.stop()

# Unpack results
cum_rets = results["cum_rets"]
daily_rets = results["daily_rets"]
signal = results["signal"]
sharpe = results["sharpe"]
max_daily_val = results["max_daily_val"]
max_daily_date = results["max_daily_date"]
max_p2t = results["max_p2t"]

# Compute 15‑day hit ratio
last_15_rets = daily_rets.tail(15)
hit_ratio = (last_15_rets > 0).sum() / len(last_15_rets) if len(last_15_rets) > 0 else 0

# --- TOP PREDICTION (big font) ---
st.markdown("## Next Trading Day Prediction")
col1, col2 = st.columns(2)
with col1:
    last_signal = signal[-1] if len(signal) > 0 else "N/A"
    st.markdown(f"<h2 style='color: #1E3A8A;'>{last_signal}</h2>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<h3 style='color: #4B5563;'>{next_open}</h3>", unsafe_allow_html=True)

st.divider()

# --- KEY METRICS (4 boxes) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sharpe Ratio", f"{sharpe:.2f}")
m2.metric("Max Daily DD", f"{max_daily_val:.2%}", max_daily_date)
m3.metric("Peak-to-Trough DD", f"{max_p2t:.2%}")
m4.metric("15d Hit Ratio", f"{hit_ratio:.0%}")

st.divider()

# --- EQUITY CURVE ---
st.subheader(f"Equity Curve: {model_name} Strategy vs. Benchmarks")
fig = go.Figure()

# Strategy
fig.add_trace(go.Scatter(
    x=cum_rets.index, y=cum_rets.values,
    name=f"{model_name} Strategy", line=dict(color="#1E3A8A")
))

# Benchmark: equal‑weight portfolio of the five ETFs
if not df.empty:
    asset_returns = df[["TLT", "TBT", "VNQ", "SLV", "GLD"]].pct_change().fillna(0)
    ew_rets = asset_returns.mean(axis=1)
    ew_rets = ew_rets.reindex(cum_rets.index).fillna(0)
    ew_cum = (1 + ew_rets).cumprod()
    fig.add_trace(go.Scatter(
        x=ew_cum.index, y=ew_cum.values,
        name="Equal Weight Benchmark", line=dict(color="#94A3B8", dash='dash')
    ))

fig.update_layout(height=400, template="plotly_white",
                  xaxis_title="Date", yaxis_title="Cumulative Return")
st.plotly_chart(fig, use_container_width=True)

# --- AUDIT TRAIL (last 15 sessions) ---
st.subheader("📋 Last 15 Sessions Audit")
audit_dates = daily_rets.tail(15).index.strftime('%Y-%m-%d').tolist()
audit_signals = signal[-15:].tolist() if len(signal) >= 15 else signal.tolist()
audit_returns = daily_rets.tail(15).values.tolist()

while len(audit_signals) < 15:
    audit_signals.insert(0, "N/A")
    audit_dates.insert(0, "N/A")
    audit_returns.insert(0, 0.0)

audit_trail = pd.DataFrame({
    "Date": audit_dates,
    "ETF Predicted": audit_signals,
    "Strategy Return (%)": [f"{r*100:.2f}%" for r in audit_returns]
})
st.table(audit_trail.iloc[::-1])

st.divider()

# --- METHODOLOGY SECTION ---
with st.expander("📘 Methodology: Transformer vs TFT"):
    st.markdown("""
    **Transformer (Attention‑based Model)**  
    A standard transformer architecture adapted for time series forecasting. It uses self‑attention to capture long‑range dependencies and is trained to predict the next day's return for each ETF. The asset with the highest predicted return is selected for the next trading day.

    **TFT (Temporal Fusion Transformer)**  
    An attention‑based model designed for interpretable multi‑horizon forecasting. It combines recurrent layers for local processing with self‑attention for long‑term dependencies, and includes built‑in mechanisms for handling known inputs (e.g., date features). Here, TFT similarly predicts next‑day returns to choose the best ETF.

    Both models are trained in a **walk‑forward** manner: for each day, only data up to that day is used to forecast the next day, simulating a realistic trading environment. Signals are precomputed offline (via GitHub Actions) and served to this dashboard.
    """)
