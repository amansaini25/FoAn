import streamlit as st
import pandas as pd
import warnings
from utils.helpers import load_global_css
from utils.data_loader import load_statsbomb_data, preprocess_passes
from engine.xt_model import apply_xt_to_passes, ExpectedThreat, prepare_xt_data
from engine.metrics import get_network_metrics, calculate_team_dna
from components.sidebar import render_data_selection, render_analysis_controls
from components.visuals import plot_passing_network, plot_top_xt, plot_zone_activity, plot_threat_pulse, plot_xt_grid
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
    page_title="FoAn Tactical DNA | Championship DNA",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load global styles
load_global_css("assets/style.css")

st.title("🏆 Championship Blueprint: Network Identity Dashboard")

# ==========================================
# 2. DATA SELECTION (SIDEBAR)
# ==========================================
selected_comp_name, selected_season_name, selected_team, team_matches = render_data_selection()

if selected_comp_name and selected_season_name and selected_team:
    st.markdown(f"### Benchmarking Tactical Connectivity: {selected_team} ({selected_season_name})")
else:
    st.markdown("### Benchmarking Tactical Connectivity")

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================
logger.info("Fetching StatsBomb data...")

if team_matches is None:
    st.info("Please select data from the sidebar to continue.")
    # render empty analysis controls
    render_analysis_controls(None)
    st.stop()
elif team_matches.empty:
    st.info("No matches found for the selected team.")
    render_analysis_controls(None)
    st.stop()
else:
    import os
    checkpoint_path = "assets/xt_checkpoint.npy"
    xt_model = None
    
    if os.path.exists(checkpoint_path):
        logger.info("Loading xT model from checkpoint...")
        xt_model = ExpectedThreat.load_checkpoint(checkpoint_path)
    
    if xt_model is None:
        with st.spinner("Fitting xT Model on all available matches (this may take a while)..."):
            logger.info("Fetching whole dataset for xT model training...")
            training_raw_df = load_statsbomb_data(team_matches, selected_team, limit_matches=None, filter_team=False)
            
            if training_raw_df.empty:
                logger.error("Failed to load training data. The dataframe is empty.")
                st.error("Failed to load training data. Please check connection.")
                render_analysis_controls(None)
                st.stop()
                
            logger.info("Preparing data for xT model and fitting...")
            actions_df = prepare_xt_data(training_raw_df)
            xt_model = ExpectedThreat(l=12, w=8, eps=1e-5)
            xt_model.fit(actions_df)
            
            if not os.path.exists("assets"):
                os.makedirs("assets", exist_ok=True)
            xt_model.save_checkpoint(checkpoint_path)
            logger.info("xT model successfully fitted and checkpoint saved.")

    # Load data for dashboard visualization (limit for speed)
    raw_df = load_statsbomb_data(team_matches, selected_team, limit_matches=5, filter_team=False)

    if raw_df.empty:
        logger.error("Failed to load dashboard data. The dataframe is empty.")
        st.error("Failed to load data. Please check your internet connection or StatsBomb API status.")
        render_analysis_controls(None)
        st.stop()
    else:
        logger.info(f"Successfully loaded {len(raw_df)} events for dashboard.")

    team_raw_df = raw_df[raw_df['team'] == selected_team].copy()
    pass_df = preprocess_passes(team_raw_df)
    pass_df = apply_xt_to_passes(pass_df, xt_model=xt_model)
    logger.info("Data processing complete, rendering dashboard...")

# ==========================================
# 4. ANALYSIS CONTROLS (SIDEBAR)
# ==========================================
filtered_df, min_pass_count, selected_time = render_analysis_controls(pass_df)

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["📊 Network Identity", "🗺️ xT Evaluation Grid"])

with tab1:
    # --- ROW 1: NETWORK HEALTH METRICS ---
    st.subheader("📊 Network Health Metrics")

    if not filtered_df.empty:
        curr_cent, curr_coh, curr_edges = get_network_metrics(filtered_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pass Volume", len(filtered_df))
        c2.metric("Centralization (Std Dev)", f"{curr_cent:.3f}", help="High = Reliance on star players. Low = Distributed.")
        c3.metric("Triadic Cohesion", f"{curr_coh:.3f}", help="High = Strong local support triangles.")
        c4.metric("Active Connections", curr_edges)

        # Team DNA Saving logic
        st.markdown("---")
        if st.button(f"💾 Save {selected_team} DNA Profile"):
            import json
            import os
            
            # 1. Prepare comprehensive df from pass_df
            comp_df = pass_df.copy()
            
            # Add time_bin
            bins = [0, 15, 30, 45, 60, 75, 90, 120]
            labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
            comp_df['time_bin'] = pd.cut(comp_df['minute'], bins=bins, labels=labels, right=False)
            
            # Add venue mapping
            if 'match_id' in comp_df.columns and team_matches is not None:
                home_matches = team_matches[team_matches['home_team'] == selected_team]['match_id'].tolist()
                comp_df['venue'] = comp_df['match_id'].apply(lambda mx: 'Home' if mx in home_matches else 'Away')
            else:
                comp_df['venue'] = 'Unknown'
                
            # Build Comprehensive Profile
            dna_comprehensive = {
                "overall": calculate_team_dna(comp_df)
            }
            
            # Outcomes
            dna_comprehensive["by_outcome"] = {}
            if 'outcome_result' in comp_df.columns:
                for outcome in comp_df['outcome_result'].dropna().unique():
                    dna_comprehensive["by_outcome"][str(outcome)] = calculate_team_dna(comp_df[comp_df['outcome_result'] == outcome])
            
            # Time Phases
            dna_comprehensive["by_time_phase"] = {}
            for phase in labels:
                phase_df = comp_df[comp_df['time_bin'] == phase]
                if not phase_df.empty:
                    dna_comprehensive["by_time_phase"][str(phase)] = calculate_team_dna(phase_df)
            
            # Venue
            dna_comprehensive["by_venue"] = {}
            for venue in ['Home', 'Away']:
                v_df = comp_df[comp_df['venue'] == venue]
                if not v_df.empty:
                    dna_comprehensive["by_venue"][str(venue)] = calculate_team_dna(v_df)
            
            # team_wise folders -> season files
            save_dir = os.path.join("team_dna", selected_team.replace(" ", "_"))
            os.makedirs(save_dir, exist_ok=True)
            
            safe_season = selected_season_name.replace("/", "_")
            file_path = os.path.join(save_dir, f"{safe_season}_dna.json")
            with open(file_path, "w") as f:
                json.dump(dna_comprehensive, f, indent=4)
                
            st.success(f"Comprehensive Team DNA saved successfully to `{file_path}`")


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
    else:
        st.warning("No pass data available for the selected filters.")

with tab2:
    plot_xt_grid(xt_model)
