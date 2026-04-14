import streamlit as st
import pandas as pd
import warnings
from utils.helpers import load_global_css
from utils.data_loader import load_statsbomb_data, preprocess_passes
from engine.xt_model import apply_xt_to_passes, ExpectedThreat, prepare_xt_data
from engine.transgoalnet import train_transgoalnet, prepare_transgoalnet_dataset, apply_transgoalnet_inference
from engine.metrics import get_network_metrics, calculate_team_dna, calculate_championship_leaderboard, generate_and_save_comprehensive_dna
from components.sidebar import render_data_selection, render_analysis_controls
from components.visuals import plot_passing_network, plot_top_xt, plot_zone_activity, plot_threat_pulse, plot_xt_grid, plot_dna_radar, plot_tactical_heatmap, plot_championship_leaderboard
from utils.logger import get_logger
import config
import os

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
load_global_css(config.STYLE_CSS)

st.title("🏆 Championship Blueprint: Network Identity Dashboard")

# ==========================================
# 2. DATA SELECTION (SIDEBAR)
# ==========================================
selected_comp_name, selected_season_name, selected_team, team_matches, all_matches, comp_id = render_data_selection()

if selected_comp_name and selected_season_name and selected_team:
    st.markdown(f"### Benchmarking Tactical Connectivity: {selected_team} ({selected_season_name})")
else:
    st.markdown("### Benchmarking Tactical Connectivity")

# ==========================================
# 2.5 GLOBAL MODEL TRAINING (SIDEBAR)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🌍 Global Model Training")

import json
import time
import subprocess

progress_file = os.path.join(config.LOGS_DIR, "training_progress.json")

# Check if currently running
is_running = False
if os.path.exists(progress_file):
    try:
        with open(progress_file, "r") as f:
            prog_data = json.load(f)
        if prog_data.get("status") == "running":
            is_running = True
            st.sidebar.info("Training is currently running in the background...")
            st.sidebar.progress(prog_data.get("progress", 0.0))
            st.sidebar.text(prog_data.get("message", "Working..."))
            time.sleep(2)
            st.rerun()
        elif prog_data.get("status") == "completed":
            st.sidebar.success("Global models successfully trained and ready to use!")
        elif prog_data.get("status") == "error":
            st.sidebar.error(f"Training failed: {prog_data.get('message')}")
    except Exception as e:
        pass

if not is_running:
    if st.sidebar.button("Start Global Training Process"):
        # Initialize progress file
        with open(progress_file, "w") as f:
            json.dump({"status": "running", "progress": 0.05, "message": "Initializing background training script..."}, f)
        
        # Spawn background process
        try:
            script_path = os.path.join(os.path.dirname(__file__), "scripts", "train_all_models.py")
            subprocess.Popen(["conda", "run", "-n", "football", "python", script_path])
            st.rerun()
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            logger.error(f"Failed to spawn background training:\n{err_msg}")
    # Global Evaluation Block
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Global Model Evaluation")
    eval_progress_file = os.path.join(config.LOGS_DIR, "evaluation_progress.json")
    
    is_eval_running = False
    if os.path.exists(eval_progress_file):
        try:
            with open(eval_progress_file, "r") as f:
                eval_data = json.load(f)
            if eval_data.get("status") == "running":
                is_eval_running = True
                st.sidebar.info("Evaluation is currently running in the background...")
                st.sidebar.progress(eval_data.get("progress", 0.0))
                st.sidebar.text(eval_data.get("message", "Working..."))
                time.sleep(2)
                st.rerun()
            elif eval_data.get("status") == "completed":
                st.sidebar.success("Global evaluation successfully compiled!")
            elif eval_data.get("status") == "error":
                st.sidebar.error(f"Evaluation failed: {eval_data.get('message')}")
        except Exception as e:
            pass

    if not is_eval_running and not is_running:
        if st.sidebar.button("Run Global Evaluation (20% Local Hold-out)"):
            with open(eval_progress_file, "w") as f:
                json.dump({"status": "running", "progress": 0.05, "message": "Initializing background evaluation script..."}, f)
            try:
                eval_script_path = os.path.join(os.path.dirname(__file__), "scripts", "evaluate_all_models.py")
                subprocess.Popen(["conda", "run", "-n", "football", "python", eval_script_path])
                st.rerun()
            except Exception as e:
                import traceback
                st.sidebar.error(f"Failed to start evaluation process: {e}")

    global_eval_file = os.path.join(config.LOGS_DIR, "global_tgn_eval.md")
    if st.sidebar.button("📊 Toggle Global Evaluation Report"):
        st.session_state['show_global_eval'] = not st.session_state.get('show_global_eval', False)
        
    if st.session_state.get('show_global_eval', False):
        if os.path.exists(global_eval_file):
            with st.sidebar.expander("Holdout 30% Test Set Results", expanded=True):
                try:
                    with open(global_eval_file, "r") as f:
                        st.markdown(f.read())
                except Exception:
                    st.error("Could not read the global evaluation file.")
        else:
            st.sidebar.info("No global evaluation report found. Please run the Global Training or Evaluation Process first.")

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
    checkpoint_path = config.XT_CHECKPOINT
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
            xt_model = ExpectedThreat(l=config.XT_L, w=config.XT_W, eps=config.XT_EPS)
            xt_model.fit(actions_df)
            
            xt_model.save_checkpoint(checkpoint_path)
            logger.info("xT model successfully fitted and checkpoint saved.")

    trans_checkpoint_path = config.TGN_GLOBAL_CHECKPOINT if os.path.exists(config.TGN_GLOBAL_CHECKPOINT) else config.TGN_CHECKPOINT
    if not os.path.exists(trans_checkpoint_path):
        import torch
        with st.spinner("Training TransGoalNet Model on GPU..."):
            logger.info("Preparing TransGoalNet Dataset...")
            actions_df = prepare_xt_data(training_raw_df) if 'training_raw_df' in locals() else prepare_xt_data(load_statsbomb_data(team_matches, selected_team, limit_matches=None, filter_team=False))
            graphs, max_n = prepare_transgoalnet_dataset(actions_df, xt_model)
            
            logger.info("Starting TransGoalNet training...")
            trans_model = train_transgoalnet(
                graphs, max_n, 
                epochs=config.TGN_EPOCHS, 
                batch_size=config.TGN_BATCH_SIZE, 
                lr=config.TGN_LR, 
                device=config.DEVICE
            )
            torch.save(trans_model.state_dict(), trans_checkpoint_path)
            logger.info("TransGoalNet successfully trained and checkpoint saved.")

    # Load data for dashboard visualization
    raw_df = load_statsbomb_data(team_matches, selected_team, limit_matches=None, filter_team=False)

    if raw_df.empty:
        logger.error("Failed to load dashboard data. The dataframe is empty.")
        st.error("Failed to load data. Please check your internet connection or StatsBomb API status.")
        render_analysis_controls(None)
        st.stop()
    else:
        logger.info(f"Successfully loaded {len(raw_df)} events for dashboard.")

    team_raw_df = raw_df[raw_df['team'] == selected_team].copy()

    passes_file = os.path.join(config.DATA_DIR, f"{selected_team.replace(' ', '_')}_saved_passes.csv")
    if os.path.exists(passes_file):
        pass_df = pd.read_csv(passes_file)
    else:       
        pass_df = preprocess_passes(team_raw_df)
        # Save processed dataframe locally
        pass_df.to_csv(passes_file, index=False)
        logger.info(f"Saved processed pass data to {passes_file}")

    pass_df = apply_xt_to_passes(pass_df, xt_model=xt_model)
    
    with st.spinner("Calculating TransGoalNet xT (Player Contributions)..."):
        pass_df, top_lane = apply_transgoalnet_inference(pass_df, basic_xt_model=xt_model, model_checkpoint_path=trans_checkpoint_path)
        
    with st.spinner("Compiling and saving Team DNA Profile silently..."):
        generate_and_save_comprehensive_dna(pass_df, team_matches, selected_team, selected_comp_name, selected_season_name, config.DNA_DIR)
            
    logger.info("Data processing complete, rendering dashboard...")

# ==========================================
# 4. ANALYSIS CONTROLS (SIDEBAR)
# ==========================================
filtered_df, min_pass_count, selected_time = render_analysis_controls(pass_df)

# ==========================================
# 4.5 PLAYER NUMBER MAPPING
# ==========================================
mapping_file = os.path.join(config.DATA_DIR, f"{selected_team.replace(' ', '_')}_player_numbers.csv")
if os.path.exists(mapping_file):
    player_mapping_df = pd.read_csv(mapping_file)
else:
    # Generate mapping from pass_df
    unique_players = pass_df['player_name'].dropna().unique()
    player_mapping_df = pd.DataFrame({
        'Player Name': unique_players,
        'Number': range(1, len(unique_players) + 1)
    })
    player_mapping_df.to_csv(mapping_file, index=False)


# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Network Identity", "🗺️ xT Evaluation Grid", "🔥 Tactical Heatmap", "🏆 Championship Leaderboard"])

with tab1:
    if not filtered_df.empty:
        # --- TEAM DNA RADAR ---
        st.subheader("🧬 Team DNA Radar (Average Match Profile)")
        
        # Calculate DNA using the full pass_df (representing whole matches)
        overall_dna_metrics = calculate_team_dna(pass_df)
        
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            plot_dna_radar(overall_dna_metrics)

        # --- ROW 1: NETWORK HEALTH METRICS ---
        st.markdown("---")
        st.subheader("📊 Network Health Metrics (Current Filter)")

        curr_cent, curr_coh, curr_edges = get_network_metrics(filtered_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pass Volume", len(filtered_df))
        c2.metric("Centralization (Std Dev)", f"{curr_cent:.3f}", help="High = Reliance on star players. Low = Distributed.")
        c3.metric("Triadic Cohesion", f"{curr_coh:.3f}", help="High = Strong local support triangles.")
        c4.metric("Active Connections", curr_edges)

        # Evaluate TransGoalNet Model
        st.markdown("---")
        
        if st.button(f"🔬 Evaluate TransGoalNet"):
            with st.spinner("Computing TransGoalNet evaluation metrics..."):
                from engine.transgoalnet import evaluate_transgoalnet
                from engine.metrics import generate_model_evaluation_report
                eval_metrics = evaluate_transgoalnet(pass_df, xt_model, trans_checkpoint_path)
                
                save_dir = os.path.join(config.LOGS_DIR, f"{selected_team.replace(' ', '_')}_tgn_eval.md")
                os.makedirs(config.LOGS_DIR, exist_ok=True)
                report_md = generate_model_evaluation_report(eval_metrics, save_dir)
                
                st.success(f"Evaluation Report saved to `{save_dir}`")
                with st.expander("View Evaluation Report", expanded=True):
                    st.markdown(report_md)


        # --- ROW 2: VISUALIZATIONS ---
        st.markdown("---")
        
        # Map player names to numbers for the passing network
        network_df = filtered_df.copy()
        mapping_dict = dict(zip(player_mapping_df['Player Name'], player_mapping_df['Number']))
        mapping_dict_str = {k: str(v) for k, v in mapping_dict.items()}
        
        network_df['player_name'] = network_df['player_name'].map(mapping_dict_str).fillna(network_df['player_name'])
        network_df['pass_recipient_name'] = network_df['pass_recipient_name'].map(mapping_dict_str).fillna(network_df['pass_recipient_name'])

        col_viz, col_mapping = st.columns([0.7, 0.3])

        with col_viz:
            plot_passing_network(network_df, min_pass_count)

        with col_mapping:
            st.subheader("🔢 Player Mapping")
            st.dataframe(player_mapping_df.set_index('Number'), use_container_width=True)

        # --- ROW 3: CRITICAL NODES & ZONE ACTIVITY ---
        st.markdown("---")
        col_zone, col_crit = st.columns([0.7, 0.3])
        
        with col_zone:
            plot_zone_activity(filtered_df)
            
        with col_crit:
            plot_top_xt(filtered_df)

        # --- ROW 4: THREAT PULSE ---
        plot_threat_pulse(pass_df, filtered_df)
    else:
        st.warning("No pass data available for the selected filters.")

with tab2:
    plot_xt_grid(xt_model)

with tab3:
    plot_tactical_heatmap(filtered_df, top_lane)

with tab4:
    scope = st.radio("Leaderboard Scope:", ["Current Season", "All-Time (All Seasons)"], horizontal=True)
    
    if scope == "Current Season":
        if all_matches is not None and not all_matches.empty:
            with st.spinner("Compiling Season Leaderboard..."):
                leaderboard_df = calculate_championship_leaderboard(
                    all_matches, 
                    selected_comp_name, 
                    selected_season_name, 
                    config.DNA_DIR, 
                    xt_model=xt_model,
                    trans_checkpoint_path=trans_checkpoint_path
                )
                plot_championship_leaderboard(leaderboard_df)
                
                st.markdown("---")
                st.subheader("🧬 Batch Visualise & Export Team DNA")
                st.write("Generate and securely save the DNA Radar plots for every team in the ranked leaderboard, or load existing ones instantly.")
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    btn_export = st.button("Generate & Export Radars", key="btn_batch_export")
                with col_b2:
                    btn_load = st.button("Load Saved Radars", key="btn_batch_load")
                    
                if btn_export or btn_load:
                    import json
                    safe_comp = selected_comp_name.replace("/", "_").replace(" ", "_")
                    safe_season = selected_season_name.replace("/", "_").replace(" ", "_")
                    
                    st.success(f"Processing {len(leaderboard_df)} teams...")
                    progress_text = "Saving radars locally..." if btn_export else "Loading saved radars..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # Columns to render neatly
                    cols = st.columns(3)
                    
                    for idx, row in leaderboard_df.iterrows():
                        team = row['Team']
                        safe_team = team.replace("/", "_").replace(" ", "_")
                        profile_path = os.path.join(config.DNA_DIR, safe_comp, safe_season, safe_team, "dna_profile.json")
                        save_path = os.path.join(config.DNA_DIR, safe_comp, safe_season, safe_team, f"{safe_team}_radar.png")
                        
                        col = cols[idx % 3] # Distribute 3 per row
                        
                        if btn_load:
                            if os.path.exists(save_path):
                                with col:
                                    st.markdown(f"**#{idx+1}. {team}**")
                                    st.image(save_path, use_container_width=True)
                            else:
                                with col:
                                    st.warning(f"No saved radar for {team}")
                        elif btn_export:
                            if os.path.exists(profile_path):
                                with open(profile_path, "r") as f:
                                    data = json.load(f)
                                    overall = data.get("overall", {})
                                    if overall:
                                        with col:
                                            st.markdown(f"**#{idx+1}. {team}**")
                                            plot_dna_radar(overall, save_path=save_path, cdi=row.get('CDI'))
                                    else:
                                        with col:
                                            st.warning(f"No DNA data for {team}")
                            else:
                                with col:
                                    st.warning(f"No profile found: {team}")
                                
                        my_bar.progress((idx + 1) / len(leaderboard_df), text=f"Processed #{idx+1} {team}")
                        
                    if btn_export:
                        st.success(f"Successfully processed all {len(leaderboard_df)} radar plots. Check your local `{config.DNA_DIR}` directory!")
                    elif btn_load:
                        st.success("Finished loading saved radars.")
                
                # MLR Optimization UI for Current Season
                st.markdown("---")
                col_mlr1, col_mlr2 = st.columns([0.7, 0.3])
                with col_mlr1:
                    st.markdown("### 🧠 Advanced TES Weighting (MLR)")
                    st.write("Train a Multiple Linear Regression model to find optimal metric weights based on this season's win ratios.")
                with col_mlr2:
                    if st.button("Optimize Weights (Current Season)", key="btn_mlr_curr"):
                        from engine.metrics import train_tes_mlr_weights
                        weights_path = os.path.join(config.DNA_DIR, selected_comp_name.replace("/", "_").replace(" ", "_"), selected_season_name.replace("/", "_").replace(" ", "_"), "tes_mlr_weights.json")
                        try:
                            # Note: Create directory if it doesn't exist
                            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                            new_weights = train_tes_mlr_weights(leaderboard_df, weights_path)
                            st.success("Weights optimized successfully!")
                            st.json(new_weights)
                            import time
                            time.sleep(2) # Give user a moment to see before reloading
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to optimize weights: {e}")
        else:
            st.info("No matches available to compute the season leaderboard.")
    else:
        st.info(f"The All-Time Leaderboard aggregates historical match results and DNA profiles across **all available seasons** for **{selected_comp_name}**.")
        if st.button("Load / Generate All-Time Leaderboard"):
            # Set a session state flag so that the dataframe persists explicitly without hiding when re-rendering components
            st.session_state['show_all_time'] = True
            
        if st.session_state.get('show_all_time', False):
            with st.spinner("Computing All-Time Leaderboard (may take a moment if not cached)..."):
                from utils.data_loader import get_competitions, get_matches
                from engine.metrics import calculate_all_time_leaderboard
                
                leaderboard_df = calculate_all_time_leaderboard(
                    selected_comp_name, 
                    comp_id, 
                    get_matches, 
                    get_competitions,
                    config.DNA_DIR, 
                    config.LEADERBOARD_DIR,
                    xt_model=xt_model,
                    trans_checkpoint_path=trans_checkpoint_path
                )
                
                if not leaderboard_df.empty:
                    plot_championship_leaderboard(leaderboard_df)
                    
                    # MLR Optimization UI for All-Time
                    st.markdown("---")
                    col_mlr1, col_mlr2 = st.columns([0.7, 0.3])
                    with col_mlr1:
                        st.markdown("### 🧠 Advanced TES Weighting (All-Time MLR)")
                        st.write("Train an MLR model targeting all-time performance statistics across all seasons for this competition.")
                    with col_mlr2:
                        if st.button("Optimize Weights (All-Time)", key="btn_mlr_alltime"):
                            from engine.metrics import train_tes_mlr_weights
                            weights_path = os.path.join(config.DNA_DIR, selected_comp_name.replace("/", "_").replace(" ", "_"), "all_time_tes_mlr_weights.json")
                            try:
                                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                                new_weights = train_tes_mlr_weights(leaderboard_df, weights_path)
                                st.success("All-Time weights optimized successfully! Generating Leaderboard using new weights...")
                                
                                # Invalidate cache so it recalculates using new weights
                                cache_file = os.path.join(config.LEADERBOARD_DIR, f"{selected_comp_name.replace('/', '_').replace(' ', '_')}_all_seasons.csv")
                                if os.path.exists(cache_file):
                                    os.remove(cache_file)
                                    
                                import time
                                time.sleep(2)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to optimize weights: {e}")
                else:
                    st.error("Failed to generate the All-Time Leaderboard.")
