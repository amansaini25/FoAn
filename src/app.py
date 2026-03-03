import streamlit as st
import pandas as pd
import warnings
from utils.helpers import load_global_css
from utils.data_loader import load_statsbomb_data, preprocess_passes
from engine.xt_model import apply_xt_to_passes, ExpectedThreat, prepare_xt_data
from engine.metrics import get_network_metrics
from components.sidebar import render_sidebar
from components.visuals import plot_passing_network, plot_top_xt, plot_zone_activity, plot_threat_pulse
from utils.logger import get_logger

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize logger
logger = get_logger()
logger.info("Initializing Championship Blueprint Dashboard...")

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="HFC Tactical DNA | Championship Blueprint",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load global styles
load_global_css("assets/style.css")

st.title("🏆 Championship Blueprint: Network Identity Dashboard")
st.markdown("### Benchmarking Tactical Connectivity: Hyderabad FC (2021/22)")

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
logger.info("Fetching StatsBomb data...")
raw_df = load_statsbomb_data(team_name="Hyderabad", limit_matches=5, filter_team=False)

if raw_df.empty:
    logger.error("Failed to load data. The dataframe is empty.")
    st.error("Failed to load data. Please check your internet connection or StatsBomb API status.")
    st.stop()
else:
    logger.info(f"Successfully loaded {len(raw_df)} events.")

with st.spinner("Fitting xT Model..."):
    logger.info("Preparing data for xT model and fitting...")
    actions_df = prepare_xt_data(raw_df)
    xt_model = ExpectedThreat(l=12, w=8, eps=1e-5)
    xt_model.fit(actions_df)
    logger.info("xT model successfully fitted.")

team_raw_df = raw_df[raw_df['team'] == "Hyderabad"].copy()
pass_df = preprocess_passes(team_raw_df)
pass_df = apply_xt_to_passes(pass_df, xt_model=xt_model)
logger.info("Data processing complete, rendering dashboard...")

# ==========================================
# 3. SIDEBAR & FILTERS
# ==========================================
filtered_df, min_pass_count, selected_time = render_sidebar(pass_df)

# ==========================================
# 4. DASHBOARD LAYOUT
# ==========================================

# --- ROW 1: NETWORK HEALTH METRICS ---
st.subheader("📊 Network Health Metrics")

curr_cent, curr_coh, curr_edges = get_network_metrics(filtered_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pass Volume", len(filtered_df))
c2.metric("Centralization (Std Dev)", f"{curr_cent:.3f}", help="High = Reliance on star players. Low = Distributed.")
c3.metric("Triadic Cohesion", f"{curr_coh:.3f}", help="High = Strong local support triangles.")
c4.metric("Active Connections", curr_edges)

# --- ROW 2: VISUALIZATIONS ---
st.markdown("---")
col_viz, col_detail = st.columns([2, 1])

with col_viz:
    plot_passing_network(filtered_df, min_pass_count)

with col_detail:
    plot_top_xt(filtered_df)
    plot_zone_activity(filtered_df)

# --- ROW 3: THREAT PULSE ---
plot_threat_pulse(pass_df, filtered_df)
