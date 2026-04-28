import streamlit as st
import pandas as pd
import warnings
from utils.helpers import load_global_css
from utils.data_loader import load_statsbomb_data, preprocess_passes
from engine.xt_model import apply_xt_to_passes, ExpectedThreat, prepare_xt_data
from engine.transgoalnet import train_transgoalnet, prepare_transgoalnet_dataset, apply_transgoalnet_inference
from engine.metrics import get_network_metrics, calculate_team_dna, calculate_championship_leaderboard, generate_and_save_comprehensive_dna
from components.sidebar import render_data_selection, render_analysis_controls
from components.visuals import plot_passing_network, plot_top_xt, plot_zone_activity, plot_threat_pulse, plot_xt_grid, plot_dna_radar, plot_dual_dna_radar, plot_tactical_heatmap, plot_championship_leaderboard
import importlib
import components.visuals
importlib.reload(components.visuals)
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
# NAVIGATION
# ==========================================
st.sidebar.markdown("---")
nav_mode = st.sidebar.radio("Dashboard Mode", ["Visual Analytics", "DNA Split Analysis", "Model Pipeline & Optimization"])

import json
import time
import subprocess

if nav_mode == "Model Pipeline & Optimization":
    st.header("⚙️ Model Pipeline & Optimization")
    st.markdown("This window isolates all heavy processing, test runs, and mathematical tuning.")
    
    st.markdown("---")
    st.subheader("📈 Model Training Telemetry")
    tcol1, tcol2, tcol3 = st.columns(3)
    
    with tcol1:
        st.markdown("**Basic xT Convergence**")
        xt_log = os.path.join(config.ASSETS_DIR, "xt_training_log.json")
        if os.path.exists(xt_log):
            import json, pandas as pd
            try:
                with open(xt_log, "r") as f:
                    xt_data = json.load(f)
                xt_df = pd.DataFrame(xt_data)
                if not xt_df.empty:
                    xt_df.set_index("iteration", inplace=True)
                    st.line_chart(xt_df["max_diff"])
            except Exception:
                st.info("Log format temporarily mismatched.")
        else:
            st.info("No xT training logs available.")

    with tcol2:
        st.markdown("**TransGoalNet Training Loss**")
        tgn_log = os.path.join(config.ASSETS_DIR, "transgoalnet_training_report.json")
        if os.path.exists(tgn_log):
            import json, pandas as pd
            try:
                with open(tgn_log, "r") as f:
                    tgn_data = json.load(f)
                    
                loss_history = tgn_data.get("evaluation_metrics", {}).get("loss_history", [])
                
                if loss_history:
                    tgn_df = pd.DataFrame({
                        "epoch": range(1, len(loss_history) + 1),
                        "loss": loss_history
                    })
                    tgn_df.set_index("epoch", inplace=True)
                    st.line_chart(tgn_df["loss"])
                else:
                    st.info("No loss history found in report.")
            except Exception:
                st.info("Log format temporarily mismatched.")
        else:
            st.info("No TGN training logs available.")
            
    with tcol3:
        tes_engine = st.session_state.get('tes_engine', 'Hybrid PCA-MLR')
        if tes_engine == 'Hybrid PCA-MLR':
            st.markdown("**TES PCA Principal Component Distribution**")
            log_prefix = "tes_pca_weights"
        else:
            st.markdown("**TES XGBoost SHAP Distribution**")
            log_prefix = "tes_xgboost_weights"
            
        log_path = os.path.join(config.ASSETS_DIR, f"{log_prefix}.json")
        
        if os.path.exists(log_path):
            import json, pandas as pd
            try:
                with open(log_path, "r") as f:
                    log_data = json.load(f)
                
                exclude_keys = ['explained_variance', 'mse', 'r2_score', 'r2_train', 'r2_test', 'cumulative_variance', 'n_components', 'model_architecture', 'saved_model_path']
                w_dict = {k: v for k, v in log_data.items() if k not in exclude_keys}
                w_df = pd.DataFrame([w_dict]).T
                w_df.columns = ["Weight Profile"]
                st.bar_chart(w_df, height=210)
                
                if "cumulative_variance" in log_data:
                    cvar = log_data['cumulative_variance']
                    n_comp = log_data.get('n_components', '?')
                    r2 = log_data.get('r2_score', 0)
                    st.success(f"**Trained on {n_comp} Principal Components (Cumulative Variance: {cvar:.1f}%) | R²:** {r2:.3f}")
                elif "model_architecture" in log_data and log_data["model_architecture"] == "xgboost":
                    r2_train = log_data.get('r2_train', 0)
                    r2_test = log_data.get('r2_test', 0)
                    st.success(f"**XGBoost SHAP (Early Stopped)** | Train R²: {r2_train:.3f} | Test R²: {r2_test:.3f}")
            except Exception:
                st.info("Log format temporarily mismatched.")
        else:
            st.info(f"No {tes_engine} optimization logs available.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🌍 Global Model Training")

    progress_file = os.path.join(config.LOGS_DIR, "training_progress.json")
    is_running = False
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                prog_data = json.load(f)
            if prog_data.get("status") == "running":
                is_running = True
                st.sidebar.info("Training is currently running...")
                st.sidebar.progress(prog_data.get("progress", 0.0))
                time.sleep(2)
                st.rerun()
            elif prog_data.get("status") == "completed":
                st.sidebar.success("Global models ready!")
            elif prog_data.get("status") == "error":
                st.sidebar.error(f"Training failed: {prog_data.get('message')}")
        except Exception:
            pass

    if not is_running:
        if st.sidebar.button("Start Global Training"):
            with open(progress_file, "w") as f:
                json.dump({"status": "running", "progress": 0.05, "message": "Initializing..."}, f)
            try:
                script_path = os.path.join(os.path.dirname(__file__), "scripts", "train_all_models.py")
                subprocess.Popen(["conda", "run", "-n", "football", "python", script_path])
                st.rerun()
            except Exception as e:
                st.sidebar.error("Failed to spawn training.")

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
                st.sidebar.info("Evaluation running...")
                st.sidebar.progress(eval_data.get("progress", 0.0))
                time.sleep(2)
                st.rerun()
            elif eval_data.get("status") == "completed":
                st.sidebar.success("Global evaluation compiled!")
            elif eval_data.get("status") == "error":
                st.sidebar.error("Evaluation failed.")
        except Exception:
            pass

    if not is_eval_running and not is_running:
        if st.sidebar.button("Run Global Evaluation (20% Hold-out)"):
            with open(eval_progress_file, "w") as f:
                json.dump({"status": "running", "progress": 0.05, "message": "Initializing..."}, f)
            try:
                eval_script_path = os.path.join(os.path.dirname(__file__), "scripts", "evaluate_all_models.py")
                subprocess.Popen(["conda", "run", "-n", "football", "python", eval_script_path])
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to start evaluation: {e}")

    global_eval_file = os.path.join(config.LOGS_DIR, "global_tgn_eval.md")
    if st.sidebar.button("📊 Toggle Global Report"):
        st.session_state['show_global_eval'] = not st.session_state.get('show_global_eval', False)
        
    if st.session_state.get('show_global_eval', False):
        if os.path.exists(global_eval_file):
            with st.expander("Global Evaluation Report (30% Test Set)", expanded=True):
                try:
                    with open(global_eval_file, "r") as f:
                        st.markdown(f.read())
                except Exception:
                    st.error("Could not read report.")
        else:
            st.info("No global evaluation report found.")

    st.sidebar.markdown("---")
    st.sidebar.header("Local Testing & Auth")
    if st.sidebar.button("🔬 Evaluate TransGoalNet (Local)"):
        st.session_state['run_local_eval'] = True
        
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Optimize TES Weights")
    tes_engine_choice = st.sidebar.radio("TES Engine", ["Hybrid PCA-MLR", "XGBoost (SHAP)"])
    st.session_state['tes_engine'] = tes_engine_choice
    
    tes_year_threshold = st.sidebar.number_input("Year Threshold (All-Time)", min_value=1990, max_value=2030, value=2015, step=1)
    if st.sidebar.button("Optimize (Current Season)"):
        st.session_state['run_opt_curr'] = True
    if st.sidebar.button("Optimize (All-Time)"):
        st.session_state['run_opt_all'] = True
        st.session_state['tes_year_threshold'] = tes_year_threshold
        
    st.sidebar.markdown("---")
    st.sidebar.header("🏆 Championship Leaderboard")
    if st.sidebar.button("Refresh Championship Leaderboard"):
        st.session_state['refresh_champ_ldr'] = True

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

    # Process Championship Leaderboard Refresh
    if st.session_state.get('refresh_champ_ldr', False):
        with st.spinner("Recompiling Championship Leaderboard..."):
            from engine.metrics import calculate_all_time_leaderboard
            from utils.data_loader import get_competitions, get_matches
            calculate_all_time_leaderboard(selected_comp_name, comp_id, get_matches, get_competitions, config.DNA_DIR, config.LEADERBOARD_DIR, xt_model=xt_model, trans_checkpoint_path=trans_checkpoint_path, force_refresh=True)
            st.session_state['refresh_champ_ldr'] = False
            st.sidebar.success("Leaderboard Refreshed!")
            
    # Load global leaderboard for min-max normalization
    safe_comp = selected_comp_name.replace("/", "_").replace(" ", "_")
    global_ldr_path = os.path.join(config.DNA_DIR, safe_comp, "all_time_leaderboard.csv")
    if os.path.exists(global_ldr_path):
        comp_all_time_df = pd.read_csv(global_ldr_path)
    else:
        comp_all_time_df = pd.DataFrame()

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
        generate_and_save_comprehensive_dna(pass_df, team_matches, selected_team, selected_comp_name, selected_season_name, config.DNA_DIR, team_raw_df)
            
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


if nav_mode == "Model Pipeline & Optimization" and team_matches is not None and not team_matches.empty:
    if st.session_state.get('run_local_eval', False):
        st.subheader(f"🔬 TransGoalNet Local Evaluation ({selected_team})")
        with st.spinner("Computing metrics..."):
            from engine.transgoalnet import evaluate_transgoalnet
            from engine.metrics import generate_model_evaluation_report
            eval_metrics = evaluate_transgoalnet(pass_df, xt_model, trans_checkpoint_path)
            save_dir = os.path.join(config.LOGS_DIR, f"{selected_team.replace(' ', '_')}_tgn_eval.md")
            os.makedirs(config.LOGS_DIR, exist_ok=True)
            report_md = generate_model_evaluation_report(eval_metrics, save_dir)
            st.success(f"Report saved to `{save_dir}`")
            st.markdown(report_md)
        st.session_state['run_local_eval'] = False
        
    if st.session_state.get('run_opt_all', False):
        st.subheader("🧠 Optimize TES Weights (Global Model)")
        with st.spinner("Computing Global All-Time Leaderboard..."):
            from utils.data_loader import get_competitions, get_matches
            from engine.metrics import calculate_all_time_leaderboard
            year_thresh = st.session_state.get('tes_year_threshold', 2015)
            ldr_df = calculate_all_time_leaderboard("Global", None, get_matches, get_competitions, config.DNA_DIR, config.LEADERBOARD_DIR, xt_model=xt_model, trans_checkpoint_path=trans_checkpoint_path, year_threshold=year_thresh)
            tes_engine = st.session_state.get('tes_engine', 'Hybrid PCA-MLR')
            
            if tes_engine == 'Hybrid PCA-MLR':
                from engine.metrics import train_tes_pca_weights
                weights_path = os.path.join(config.ASSETS_DIR, "tes_pca_weights.json")
            else:
                from engine.metrics import train_tes_xgboost_weights
                weights_path = os.path.join(config.ASSETS_DIR, "tes_xgboost_weights.json")
                xgb_model_path = os.path.join(config.ASSETS_DIR, "global_xgboost_model.json")
                
            try:
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                if tes_engine == 'Hybrid PCA-MLR':
                    new_weights = train_tes_pca_weights(ldr_df, weights_path)
                else:
                    new_weights = train_tes_xgboost_weights(ldr_df, weights_path, xgb_model_path)
                st.success(f"{tes_engine} Global weights optimized successfully!")
                st.json(new_weights)
            except Exception as e:
                st.error(f"Failed to optimize weights: {e}")
        st.session_state['run_opt_all'] = False


elif nav_mode == "DNA Split Analysis" and team_matches is not None and not team_matches.empty:
    st.header(f"🧬 DNA Split Analysis: {selected_team}")
    st.markdown("Comparing the team's historical average DNA against their tactical identity in explicitly won and lost matches to determine absolute TES shifts across identical global boundaries.")
    
    safe_comp = selected_comp_name.replace("/", "_").replace(" ", "_")
    safe_season = selected_season_name.replace("/", "_").replace(" ", "_")
    safe_team = selected_team.replace("/", "_").replace(" ", "_")
    profile_path = os.path.join(config.DNA_DIR, safe_comp, safe_season, safe_team, "dna_profile.json")
    
    with st.spinner("Compiling contextual splits..."):
        if os.path.exists(profile_path):
            with open(profile_path, "r") as f:
                dna_data = json.load(f)
            overall = dna_data.get("overall", {})
            by_outcome = dna_data.get("by_outcome", {})
            win_stats = by_outcome.get("Win", {})
            loss_stats = by_outcome.get("Loss", {})
            draw_stats = by_outcome.get("Draw", {})
            
            from engine.metrics import calculate_championship_leaderboard, get_tes_weights
            leaderboard_df = calculate_championship_leaderboard(all_matches, selected_comp_name, selected_season_name, config.DNA_DIR, xt_model=xt_model, trans_checkpoint_path=trans_checkpoint_path)
            
            tes_engine = st.session_state.get('tes_engine', 'Hybrid PCA-MLR')
            log_prefix = "tes_pca_weights" if tes_engine == 'Hybrid PCA-MLR' else "tes_xgboost_weights"
            weights_path = os.path.join(config.ASSETS_DIR, f"{log_prefix}.json")
                
            if os.path.exists(weights_path):
                w_coh, w_txt, w_bxt, w_dec, w_xg, w_pacc, w_ret, w_itr = get_tes_weights(weights_path)
            else:
                w_coh, w_txt, w_bxt, w_dec, w_xg, w_pacc, w_ret, w_itr = 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125
                
            def calc_tes(subset_dict):
                if not subset_dict or leaderboard_df.empty: return 0.0
                
                def get_norm(val, col):
                    if col not in leaderboard_df.columns: return 0.5
                    c_min = leaderboard_df[col].min()
                    c_max = leaderboard_df[col].max()
                    if c_max == c_min or pd.isna(c_max) or pd.isna(c_min): return 0.5
                    return (val - c_min) / (c_max - c_min)
                    
                coh_norm = get_norm(subset_dict.get('avg_cohesion', 0), 'Cohesion')
                txt_norm = get_norm(subset_dict.get('avg_trans_xt', 0), 'Trans_xT')
                bxt_norm = get_norm(subset_dict.get('avg_xt', 0), 'Basic_xT')
                dec_norm = 1.0 - get_norm(subset_dict.get('avg_centralization', 0), 'Centralization')
                xg_norm = get_norm(subset_dict.get('avg_xg', 0), 'xG')
                pacc_norm = get_norm(subset_dict.get('avg_pass_acc', 0), 'Pass_Acc')
                ret_norm = get_norm(subset_dict.get('avg_retention', 0), 'Retention')
                itr_norm = get_norm(subset_dict.get('avg_itrans', 0), 'ITrans')
                
                tes = (w_coh * coh_norm) + (w_txt * txt_norm) + (w_bxt * bxt_norm) + (w_dec * dec_norm) + (w_xg * xg_norm) + (w_pacc * pacc_norm) + (w_ret * ret_norm) + (w_itr * itr_norm)
                return max(0.0, tes * 100) # Scale TES to 0-100 for display
            
            if not leaderboard_df.empty and len(leaderboard_df) > 1 and selected_team in leaderboard_df['Team'].values:
                team_row = leaderboard_df[leaderboard_df['Team'] == selected_team].iloc[0]
                
                tes_overall = calc_tes(overall)
                tes_win = calc_tes(win_stats)
                tes_loss = calc_tes(loss_stats)
                tes_draw = calc_tes(draw_stats)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"**🛡️ Historical (TES: {tes_overall:.2f})**")
                    if overall:
                        plot_dna_radar(overall, cdi=tes_overall, leaderboard_df=leaderboard_df)
                    else: st.warning("No data")
                with col2:
                    st.markdown(f"**🏆 Wins (TES: {tes_win:.2f})**")
                    if win_stats:
                        plot_dna_radar(win_stats, cdi=tes_win, leaderboard_df=leaderboard_df)
                    else: st.warning("No wins found")
                with col3:
                    st.markdown(f"**⚠️ Losses (TES: {tes_loss:.2f})**")
                    if loss_stats:
                        plot_dna_radar(loss_stats, cdi=tes_loss, leaderboard_df=leaderboard_df)
                    else: st.warning("No losses found")
                with col4:
                    st.markdown(f"**⚖️ Draws (TES: {tes_draw:.2f})**")
                    if draw_stats:
                        plot_dna_radar(draw_stats, cdi=tes_draw, leaderboard_df=leaderboard_df)
                    else: st.warning("No draws found")
            else:
                st.warning("Could not contextualize TES against the leaderboard minimums and maximums.")
                
        else:
            st.error("Team DNA profile not found. Please navigate to the Leaderboard to Batch Render team profiles.")

        # --- DNA SIMILARITY MODULE ---
        st.markdown("---")
        st.subheader("🔍 DNA Similarity Comparison")
        
        with st.expander("📚 DNA Feature Definitions"):
            st.markdown("""
            - **Basic xT**: Traditional expected threat generated via passing.
            - **Cohesion**: Network triadic cohesion (amount of local support triangles).
            - **Decentralization**: Low dependency on individual star players for distribution (1 - Centralization from standard deviation).
            - **TransxT**: Dynamic Expected Threat calculated over a sliding temporal window of actions using our Graph Transformer.
            - **Retention**: Ability to keep possession without being dispossessed.
            - **Pass Accuracy**: Ratio of successful passes to total passes.
            - **xG**: Expected Goals generated via shots.
            - **ITrans**: Inverse Transition phase threat; expected threat over possession duration.
            """)

        sim_mode = st.radio("Comparison Mode:", ["Team vs Team", "Historical vs Match"], horizontal=True)
        
        from scipy.spatial.distance import cosine
        import numpy as np
        
        def extract_dna_vector(dna_metrics, ldf=None):
            if not dna_metrics: return None
            
            def get_norm(val, col):
                if ldf is None or ldf.empty or col not in ldf.columns:
                    # Fallback to absolute max caps
                    max_caps = {
                        'Centralization': 1.0, 'Cohesion': 0.2, 'Basic_xT': 2.0, 
                        'ITrans': 0.002, 'xG': 3.0, 'Pass_Acc': 1.0, 
                        'Retention': 20.0, 'Trans_xT': 15.0
                    }
                    m = max_caps.get(col, 1.0)
                    return max(0.01, min(val / m, 1.0))
                
                c_min = ldf[col].min()
                c_max = ldf[col].max()
                if c_max == c_min or pd.isna(c_max) or pd.isna(c_min): return 0.5
                return (val - c_min) / (c_max - c_min)

            cent = dna_metrics.get('avg_centralization', 0.0)
            decent = 1.0 - get_norm(cent, 'Centralization')
            
            values = [
                decent,
                get_norm(dna_metrics.get('avg_cohesion', 0.0), 'Cohesion'),
                get_norm(dna_metrics.get('avg_xt', 0.0), 'Basic_xT'),
                get_norm(dna_metrics.get('avg_itrans', 0.0), 'ITrans'),
                get_norm(dna_metrics.get('avg_xg', 0.0), 'xG'),
                get_norm(dna_metrics.get('avg_pass_acc', 0.0), 'Pass_Acc'),
                get_norm(dna_metrics.get('avg_retention', 0.0), 'Retention'),
                get_norm(dna_metrics.get('avg_trans_xt', 0.0), 'Trans_xT')
            ]
            return np.array(values)
            
        _ldf = leaderboard_df if 'leaderboard_df' in locals() else None
        vector_1 = extract_dna_vector(overall if 'overall' in locals() else None, _ldf)
        
        def display_dna_comparison_table(name1, dna1, name2, dna2):
            if not dna1 or not dna2: return
            categories = ['Cohesion', 'Decentralization', 'Retention', 'Pass Accuracy', 'xG', 'ITrans', 'Basic xT', 'TransxT']
            
            def extract_raw(dna):
                return [
                    dna.get('avg_cohesion', 0.0),
                    1.0 - dna.get('avg_centralization', 1.0) if dna.get('avg_centralization', 1.0) < 1.0 else 0.0,
                    dna.get('avg_retention', 0.0),
                    dna.get('avg_pass_acc', 0.0),
                    dna.get('avg_xg', 0.0),
                    dna.get('avg_itrans', 0.0),
                    dna.get('avg_xt', 0.0),
                    dna.get('avg_trans_xt', 0.0)
                ]
            
            df = pd.DataFrame({
                'Metric': categories,
                name1: extract_raw(dna1),
                name2: extract_raw(dna2)
            })
            
            # Difference column
            df['Difference'] = df[name1] - df[name2]
            
            format_dict = {
                name1: '{:.3f}',
                name2: '{:.3f}',
                'Difference': '{:+.3f}'
            }
            
            def highlight_diff(val):
                if pd.isna(val): return ''
                color = '#00ff85' if val > 0 else '#ff4b4b' if val < 0 else 'white'
                return f'color: {color}'
                
            st.markdown("### Numerical Comparison")
            st.dataframe(df.style.format(format_dict).map(highlight_diff, subset=['Difference']), use_container_width=True)
        
        if sim_mode == "Team vs Team":
            teams = pd.concat([all_matches['home_team'], all_matches['away_team']]).unique() if all_matches is not None else []
            teams = sorted([t for t in teams if t != selected_team])
            if teams:
                target_team = st.selectbox("Select Target Team", teams)
                safe_target = target_team.replace("/", "_").replace(" ", "_")
                target_profile_path = os.path.join(config.DNA_DIR, safe_comp, safe_season, safe_target, "dna_profile.json")
                
                if os.path.exists(target_profile_path):
                    with open(target_profile_path, "r") as f:
                        target_data = json.load(f)
                    target_overall = target_data.get("overall", {})
                    vector_2 = extract_dna_vector(target_overall, _ldf)
                    
                    if vector_1 is not None and vector_2 is not None:
                        sim = 1 - cosine(vector_1, vector_2)
                        st.metric(f"Cosine Similarity ({selected_team} vs {target_team})", f"{sim*100:.2f}%")
                        
                        st.markdown("### Tactical Radar Comparison")
                        tes1 = calc_tes(overall) if 'calc_tes' in locals() else None
                        tes2 = calc_tes(target_overall) if 'calc_tes' in locals() else None
                        plot_dual_dna_radar(selected_team, overall, target_team, target_overall, leaderboard_df=_ldf, tes1=tes1, tes2=tes2)
                        display_dna_comparison_table(selected_team, overall, target_team, target_overall)
                    else:
                        st.warning("Could not extract feature vectors.")
                else:
                    st.warning(f"DNA profile not found for {target_team}. Please batch render team profiles on the Leaderboard tab.")
            else:
                st.info("No other teams available for comparison.")
                
        elif sim_mode == "Historical vs Match":
            if team_matches is not None and not team_matches.empty:
                matches_list = team_matches.copy()
                matches_list['Match_Label'] = matches_list['home_team'] + " vs " + matches_list['away_team'] + " (" + matches_list['match_date'].astype(str) + ")"
                match_id = st.selectbox("Select Match", matches_list['match_id'].tolist(), format_func=lambda x: matches_list[matches_list['match_id'] == x]['Match_Label'].iloc[0])
                
                if st.button("Calculate Match Similarity"):
                    with st.spinner("Extracting Match DNA..."):
                        if match_id:
                            m_raw = raw_df[(raw_df['match_id'] == match_id) & (raw_df['team'] == selected_team)]
                            m_pass = pass_df[pass_df['match_id'] == match_id]
                            from engine.metrics import calculate_team_dna
                            match_dna = calculate_team_dna(m_pass, m_raw)
                            vector_2 = extract_dna_vector(match_dna, _ldf)
                            
                            if vector_1 is not None and vector_2 is not None:
                                sim = 1 - cosine(vector_1, vector_2)
                                st.metric(f"Cosine Similarity (Historical vs Selected Match)", f"{sim*100:.2f}%")
                                
                                st.markdown("### Tactical Radar Comparison")
                                tes1 = calc_tes(overall) if 'calc_tes' in locals() else None
                                tes2 = calc_tes(match_dna) if 'calc_tes' in locals() else None
                                plot_dual_dna_radar(f"{selected_team} (Historical)", overall, "Match DNA", match_dna, leaderboard_df=_ldf, tes1=tes1, tes2=tes2)
                                display_dna_comparison_table(f"{selected_team} (Historical)", overall, "Match DNA", match_dna)
                            else:
                                st.warning("Could not calculate match vectors.")
            else:
                st.info("No matches available.")

elif nav_mode == "Visual Analytics" and team_matches is not None and not team_matches.empty:
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
            overall_dna_metrics = calculate_team_dna(pass_df, team_raw_df)
            
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                plot_dna_radar(overall_dna_metrics, leaderboard_df=comp_all_time_df)

            # --- ROW 1: NETWORK HEALTH METRICS ---
            st.markdown("---")
            st.subheader("📊 Network Health Metrics (Current Filter)")

            curr_cent, curr_coh, curr_edges = get_network_metrics(filtered_df)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pass Volume", len(filtered_df))
            c2.metric("Centralization (Std Dev)", f"{curr_cent:.3f}", help="High = Reliance on star players. Low = Distributed.")
            c3.metric("Triadic Cohesion", f"{curr_coh:.3f}", help="High = Strong local support triangles.")
            c4.metric("Active Connections", curr_edges)


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
        col_scope, col_engine = st.columns(2)
        with col_scope:
            scope = st.radio("Leaderboard Scope:", ["Current Season", "All-Time (All Seasons)"], horizontal=True)
        with col_engine:
            def_idx = 0 if st.session_state.get('tes_engine', 'Hybrid PCA-MLR') == 'Hybrid PCA-MLR' else 1
            rank_engine = st.radio("TES Ranking Engine:", ["Hybrid PCA-MLR", "XGBoost (SHAP)"], index=def_idx, horizontal=True)
        
        if scope == "Current Season":
            if all_matches is not None and not all_matches.empty:
                with st.spinner("Compiling Season Leaderboard..."):
                    leaderboard_df = calculate_championship_leaderboard(
                        all_matches, 
                        selected_comp_name, 
                        selected_season_name, 
                        config.DNA_DIR, 
                        xt_model=xt_model,
                        trans_checkpoint_path=trans_checkpoint_path,
                        engine_type=rank_engine
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
                                                plot_dna_radar(overall, save_path=save_path, cdi=row.get('TES') * 100 if row.get('TES') else None)
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
                    

            else:
                st.info("No matches available to compute the season leaderboard.")
        else:
            st.info(f"The All-Time Leaderboard aggregates historical match results and DNA profiles globally across **all available seasons for {selected_comp_name}**.")
            if st.button("Load / Generate All-Time Leaderboard"):
                # Set a session state flag so that the dataframe persists explicitly without hiding when re-rendering components
                st.session_state['show_all_time'] = True
                
            if st.session_state.get('show_all_time', False):
                with st.spinner("Computing Global All-Time Leaderboard (may take a moment if not cached)..."):
                    from utils.data_loader import get_competitions, get_matches
                    from engine.metrics import calculate_all_time_leaderboard
                    
                    year_thresh = st.session_state.get('tes_year_threshold', 2015)
                    leaderboard_df = calculate_all_time_leaderboard(
                        selected_comp_name, 
                        comp_id, 
                        get_matches, 
                        get_competitions,
                        config.DNA_DIR, 
                        config.LEADERBOARD_DIR,
                        xt_model=xt_model,
                        trans_checkpoint_path=trans_checkpoint_path,
                        year_threshold=year_thresh,
                        engine_type=rank_engine
                    )
                    
                    if not leaderboard_df.empty:
                        plot_championship_leaderboard(leaderboard_df)
                        

                    else:
                        st.error("Failed to generate the All-Time Leaderboard.")
