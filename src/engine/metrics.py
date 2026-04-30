import networkx as nx
import numpy as np
import pandas as pd

def get_network_metrics(data):
    """
    Calculates network-based metrics from passing data.
    """
    G = nx.DiGraph()
    if data.empty:
        return 0, 0, 0
        
    pass_counts = data.groupby(['player_name', 'pass_recipient_name']).size().reset_index(name='weight')
    for _, row in pass_counts.iterrows():
        if row['player_name'] != row['pass_recipient_name']:
            G.add_edge(row['player_name'], row['pass_recipient_name'], weight=row['weight'])
    
    
    if len(G) == 0: return 0, 0, 0
    
    # Metrics
    bet = nx.betweenness_centrality(G, weight='weight')
    cent = np.std(list(bet.values())) # Centralization
    clus = nx.clustering(G, weight='weight')
    coh = np.mean(list(clus.values())) # Cohesion
    
    return cent, coh, len(G.edges)

def calculate_dsi(match_events, team_name):
    """
    Calculates Defensive Suppression Index (DSI) and High-Efficiency Suppression (HES)
    for a given team based on StatsBomb match events.
    """
    if match_events.empty:
        return 0.0, 0.0
        
    opp_events = match_events[match_events['team'] != team_name].copy()
    our_events = match_events[match_events['team'] == team_name].copy()
    
    if opp_events.empty:
        return 0.0, 0.0
        
    # Handle NaN in under_pressure
    if 'under_pressure' in opp_events.columns:
        opp_events['under_pressure'] = opp_events['under_pressure'].fillna(False)
    else:
        opp_events['under_pressure'] = False
        
    opp_passes = opp_events[opp_events['type'] == 'Pass'].copy()
    total_opp_passes = len(opp_passes)
    
    if total_opp_passes == 0:
        return 0.0, 0.0
        
    def get_spatial_weight(x, y):
        if pd.isna(x) or pd.isna(y): return 1.0
        # Golden Zone: box (x>=102, 18<=y<=62) and Zone 14 (80<=x<=102, 30<=y<=50)
        in_box = (x >= 102) and (18 <= y <= 62)
        in_zone14 = (80 <= x <= 102) and (30 <= y <= 50)
        if in_box or in_zone14:
            return 1.5
        return 1.0
        
    # Calculate DSI
    opp_passes['pressure_weight'] = opp_passes.apply(
        lambda row: get_spatial_weight(row.get('location_0', row.get('x')), row.get('location_1', row.get('y'))) 
        if row['under_pressure'] else 0.0, axis=1
    )
    
    opp_passes_under_pressure_weighted = opp_passes['pressure_weight'].sum()
    
    if 'pass_outcome' in opp_passes.columns:
        completed_passes = opp_passes[opp_passes['pass_outcome'].isnull()]
    else:
        completed_passes = opp_passes
        
    completion_rate = len(completed_passes) / total_opp_passes
    
    dsi = (opp_passes_under_pressure_weighted / total_opp_passes) * (1 - completion_rate)
    
    # Calculate High-Efficiency Suppression (HES)
    hes = 0.0
    if not our_events.empty:
        pressures = our_events[our_events['type'] == 'Pressure']
        recoveries = our_events[our_events['type'].isin(['Ball Recovery', 'Interception'])]
        
        for _, p_row in pressures.iterrows():
            p_time = p_row.get('minute', 0) * 60 + p_row.get('second', 0)
            p_period = p_row.get('period', 1)
            
            # Find recoveries within 3 seconds
            recs = recoveries[(recoveries['period'] == p_period) & 
                              ((recoveries['minute'] * 60 + recoveries['second']) >= p_time) & 
                              ((recoveries['minute'] * 60 + recoveries['second']) <= p_time + 3)]
            if not recs.empty:
                hes += 1.0
                
    return float(dsi), float(hes)

def get_penalized_opponent_network(m_raw_opp, dsi):
    """
    Builds the opponent network and applies 'Friction' based on DSI to penalize centrality.
    """
    if m_raw_opp.empty: return 0.0
    opp_passes = m_raw_opp[m_raw_opp['type'] == 'Pass'].copy()
    if 'pass_outcome' in opp_passes.columns:
        opp_success = opp_passes[opp_passes['pass_outcome'].isnull()]
    else:
        opp_success = opp_passes
        
    G = nx.DiGraph()
    if opp_success.empty: return 0.0
    
    # raw events have 'player' and 'pass_recipient' instead of 'player_name'
    if 'player' not in opp_success.columns or 'pass_recipient' not in opp_success.columns:
        return 0.0
        
    pass_counts = opp_success.groupby(['player', 'pass_recipient']).size().reset_index(name='weight')
    friction_coefficient = max(0.01, 1.0 - dsi) # Higher DSI = More Friction
    
    for _, row in pass_counts.iterrows():
        if row['player'] != row['pass_recipient']:
            w = row['weight'] * friction_coefficient
            G.add_edge(row['player'], row['pass_recipient'], weight=w)
            
    if len(G) == 0: return 0.0
    bet = nx.betweenness_centrality(G, weight='weight')
    return np.std(list(bet.values()))

def calculate_team_dna(df, raw_df=None):
    """
    Calculates the Team DNA metrics from the full passes dataframe, averaged per match.
    """
    if df.empty:
        return {}
        
    match_ids = df['match_id'].unique() if 'match_id' in df.columns else ['single_match']
    num_matches = len(match_ids) if len(match_ids) > 0 else 1
    
    # Per-match arrays
    volumes = []
    centralizations = []
    cohesions = []
    conns = []
    xts = []
    trans_xts = []
    
    xgs = []
    pass_accs = []
    retentions = []
    itrans_list = []
    dsis = []
    hes_list = []
    opp_cents = []
    
    for match_id in match_ids:
        # If 'match_id' isn't in df, just use the whole df
        m_df = df[df['match_id'] == match_id] if 'match_id' in df.columns else df
        m_raw = raw_df[raw_df['match_id'] == match_id] if raw_df is not None and 'match_id' in raw_df.columns else raw_df
        
        volumes.append(len(m_df))
        c, co, a = get_network_metrics(m_df)
        centralizations.append(c)
        cohesions.append(co)
        conns.append(a)
        
        m_xt = m_df['xT'].sum() if 'xT' in m_df.columns else 0.0
        xts.append(m_xt)
        
        m_trans_xt = m_df['Trans_xT'].sum() if 'Trans_xT' in m_df.columns else 0.0
        trans_xts.append(m_trans_xt)
        
        # New Metrics
        if m_raw is not None and not m_raw.empty:
            # Determine the selected team from the pass df for per-team scoping
            selected_team = df['team'].iloc[0] if 'team' in df.columns else None
            
            # Scope raw events to the selected team for per-team metrics
            # (full m_raw is still used for DSI/HES which needs opponent events)
            m_team_raw = m_raw[m_raw['team'] == selected_team] if selected_team else m_raw
            
            # xG: only the selected team's shots
            m_xg = m_team_raw['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in m_team_raw.columns else 0.0
            xgs.append(m_xg)
            
            # Pass accuracy: team passes only
            team_passes = m_team_raw[m_team_raw['type'] == 'Pass']
            total_passes = len(team_passes)
            # m_df contains successful passes (pass_outcome is null after preprocess)
            pass_acc = len(m_df) / total_passes if total_passes > 0 else 0.0
            pass_accs.append(min(pass_acc, 1.0))  # cap at 1.0 to prevent rounding errors
            
            # Retention: team's own losses / team's possession sequences
            team_failed_passes = len(team_passes[team_passes['pass_outcome'].notnull()]) if 'pass_outcome' in team_passes.columns else 0
            team_losses = team_failed_passes + len(m_team_raw[m_team_raw['type'].isin(['Dispossessed', 'Miscontrol'])])
            m_poss_count = m_team_raw['possession'].nunique() if 'possession' in m_team_raw.columns else 1.0
            retention = team_losses / m_poss_count if m_poss_count > 0 else 0.0
            retentions.append(retention)
            
            # ITrans: Trans_xT per second of possession duration (team only)
            m_poss_duration = m_team_raw['duration'].sum() if 'duration' in m_team_raw.columns else 0.0
            itrans = m_trans_xt / m_poss_duration if m_poss_duration > 0 else 0.0
            itrans_list.append(itrans)
            
            # DSI and HES use FULL m_raw (opponent events needed)
            if selected_team:
                m_dsi, m_hes = calculate_dsi(m_raw, selected_team)
                dsis.append(m_dsi)
                hes_list.append(m_hes)
                
                m_raw_opp = m_raw[m_raw['team'] != selected_team]
                opp_penalized_cent = get_penalized_opponent_network(m_raw_opp, m_dsi)
                opp_cents.append(opp_penalized_cent)
            else:
                dsis.append(0.0)
                hes_list.append(0.0)
                opp_cents.append(0.0)
                
        else:
            xgs.append(0.0)
            pass_accs.append(0.0)
            retentions.append(0.0)
            itrans_list.append(0.0)
            dsis.append(0.0)
            hes_list.append(0.0)
            opp_cents.append(0.0)
        
    avg_volume = np.mean(volumes) if volumes else 0.0
    avg_cent = np.mean(centralizations) if centralizations else 0.0
    avg_coh = np.mean(cohesions) if cohesions else 0.0
    avg_conns = np.mean(conns) if conns else 0.0
    avg_xt = np.mean(xts) if xts else 0.0
    avg_trans_xt = np.mean(trans_xts) if trans_xts else 0.0
    
    avg_xg = np.mean(xgs) if xgs else 0.0
    avg_pass_acc = np.mean(pass_accs) if pass_accs else 0.0
    avg_retention = np.mean(retentions) if retentions else 0.0
    avg_itrans = np.mean(itrans_list) if itrans_list else 0.0
    
    avg_dsi = np.mean(dsis) if dsis else 0.0
    avg_hes = np.mean(hes_list) if hes_list else 0.0
    avg_opp_cent = np.mean(opp_cents) if opp_cents else 0.0
    
    xt_per_pass = avg_xt / avg_volume if avg_volume > 0 else 0.0
    trans_xt_per_pass = avg_trans_xt / avg_volume if avg_volume > 0 else 0.0
    delta_xt = avg_trans_xt
    
    # 4. Top Threat Creators (overall across all matches for simplicity, divided by matches)
    top_creators = {}
    if 'xT' in df.columns and 'player_name' in df.columns:
        player_xt = (df.groupby('player_name')['xT'].sum() / num_matches).sort_values(ascending=False)
        top_creators = player_xt.head(3).to_dict()

    top_trans_creators = {}
    if 'Trans_xT' in df.columns and 'player_name' in df.columns:
        player_trans_xt = (df.groupby('player_name')['Trans_xT'].sum() / num_matches).sort_values(ascending=False)
        top_trans_creators = player_trans_xt.head(3).to_dict()
        
    dna_metrics = {
        "avg_pass_volume": float(avg_volume),
        "avg_active_connections": float(avg_conns),
        "avg_centralization": float(avg_cent),
        "avg_cohesion": float(avg_coh),
        "avg_xt": float(avg_xt),
        "xt_per_pass": float(xt_per_pass),
        "avg_trans_xt": float(avg_trans_xt),
        "trans_xt_per_pass": float(trans_xt_per_pass),
        "delta_xt": float(delta_xt),
        "avg_xg": float(avg_xg),
        "avg_pass_acc": float(avg_pass_acc),
        "avg_retention": float(avg_retention),
        "avg_itrans": float(avg_itrans),
        "avg_dsi": float(avg_dsi),
        "avg_hes": float(avg_hes),
        "avg_opp_cent_penalized": float(avg_opp_cent),
        "top_threat_creators": top_creators,
        "top_trans_threat_creators": top_trans_creators
    }
    
    return dna_metrics

def generate_and_save_comprehensive_dna(pass_df, team_matches, selected_team, selected_comp_name, selected_season_name, dna_dir, raw_df=None):
    """
    Parses a team's passing structure across periods, venues, and match results.
    Saves the entire compilation as a dna_profile.json format locally.
    Returns the comprehensive dictionary.
    """
    import os
    import json
    
    comp_df = pass_df.copy()
    
    # Add time_bin
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
    if not comp_df.empty and 'minute' in comp_df.columns:
        comp_df['time_bin'] = pd.cut(comp_df['minute'], bins=bins, labels=labels, right=False)
    else:
        comp_df['time_bin'] = 'Unknown'
    
    # Add venue mapping
    if 'match_id' in comp_df.columns and team_matches is not None and not team_matches.empty:
        home_matches = team_matches[team_matches['home_team'] == selected_team]['match_id'].tolist()
        comp_df['venue'] = comp_df['match_id'].apply(lambda mx: 'Home' if mx in home_matches else 'Away')
    else:
        comp_df['venue'] = 'Unknown'
        
    dna_comprehensive = {
        "overall": calculate_team_dna(comp_df, raw_df)
    }
    
    dna_comprehensive["by_outcome"] = {}
    if 'outcome_result' in comp_df.columns:
        for outcome in comp_df['outcome_result'].dropna().unique():
            dna_comprehensive["by_outcome"][str(outcome)] = calculate_team_dna(comp_df[comp_df['outcome_result'] == outcome], raw_df)
            
    dna_comprehensive["by_time_phase"] = {}
    if 'time_bin' in comp_df.columns and not comp_df.empty:
        for phase in labels:
            phase_df = comp_df[comp_df['time_bin'] == phase]
            if not phase_df.empty:
                dna_comprehensive["by_time_phase"][str(phase)] = calculate_team_dna(phase_df, raw_df)
            
    dna_comprehensive["by_venue"] = {}
    for venue in ['Home', 'Away']:
        v_df = comp_df[comp_df['venue'] == venue]
        if not v_df.empty:
            dna_comprehensive["by_venue"][str(venue)] = calculate_team_dna(v_df, raw_df)
            
    safe_comp = selected_comp_name.replace("/", "_").replace(" ", "_")
    safe_season = selected_season_name.replace("/", "_").replace(" ", "_")
    safe_team = selected_team.replace("/", "_").replace(" ", "_")
    
    save_dir = os.path.join(dna_dir, safe_comp, safe_season, safe_team)
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, "dna_profile.json")
    with open(file_path, "w") as f:
        json.dump(dna_comprehensive, f, indent=4)
        
    return dna_comprehensive

def generate_model_evaluation_report(eval_metrics, save_path):
    """
    Generates and saves a Markdown report for TransGoalNet evaluation metrics.
    """
    md_content = "# TransGoalNet Model Evaluation Report\n\n"
    md_content += "| Metric Category | Primary Tool | Value | What it tells you |\n"
    md_content += "| --- | --- | --- | --- |\n"
    
    for category, det in eval_metrics.items():
        val_str = f"{det['Value']:.5f}" if isinstance(det['Value'], float) else str(det['Value'])
        row = f"| {category} | {det['Metric']} | **{val_str}** | {det['Meaning']} |\n"
        md_content += row
        
    with open(save_path, "w") as f:
        f.write(md_content)
        
    return md_content

def get_team_match_results(matches_df):
    """
    Calculates Win/Loss/Draw ratios for all teams in a given matches dataframe.
    """
    if matches_df.empty:
        return pd.DataFrame()
        
    team_stats = {}
    
    for _, match in matches_df.iterrows():
        home = match['home_team']
        away = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        
        if home not in team_stats:
            team_stats[home] = {'W': 0, 'D': 0, 'L': 0, 'Matches': 0}
        if away not in team_stats:
            team_stats[away] = {'W': 0, 'D': 0, 'L': 0, 'Matches': 0}
            
        team_stats[home]['Matches'] += 1
        team_stats[away]['Matches'] += 1
        
        if home_score > away_score:
            team_stats[home]['W'] += 1
            team_stats[away]['L'] += 1
        elif home_score < away_score:
            team_stats[home]['L'] += 1
            team_stats[away]['W'] += 1
        else:
            team_stats[home]['D'] += 1
            team_stats[away]['D'] += 1
            
    # Convert to DataFrame
    rows = []
    for team, stats in team_stats.items():
        w_r = stats['W'] / stats['Matches'] if stats['Matches'] > 0 else 0
        l_r = stats['L'] / stats['Matches'] if stats['Matches'] > 0 else 0
        spread = w_r - l_r
        rows.append({
            'Team': team,
            'Matches': stats['Matches'],
            'Wins': stats['W'],
            'Draws': stats['D'],
            'Losses': stats['L'],
            'Win_Ratio': w_r,
            'Loss_Ratio': l_r,
            'WL_Spread': spread
        })
        
    df = pd.DataFrame(rows)
    # Normalize spread min-max
    if len(df) > 1 and df['WL_Spread'].max() != df['WL_Spread'].min():
        df['Spread_Norm'] = (df['WL_Spread'] - df['WL_Spread'].min()) / (df['WL_Spread'].max() - df['WL_Spread'].min())
    else:
        df['Spread_Norm'] = 0.5
        
    return df

def train_tes_pca_weights(save_path):
    """
    Trains a Hybrid PCA-MLR model from the match-level features
    to extract variance dynamically and bind it onto match_goal_difference.
    """
    import json
    import numpy as np
    import os
    import config
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    train_path = os.path.join(config.DATA_DIR, "tes_train_data.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("Match-level training data not found. Please run src/scripts/build_tes_dataset.py first.")
        
    df = pd.read_csv(train_path)
    features = ['Coh', 'TxT', 'BxT', 'Dec', 'xG', 'PAcc', 'Ret', 'ITr', 'DSI', 'HES']
    
    # Standardize features within each competition and season
    df[features] = df.groupby(['competition_id', 'season_id'])[features].transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))
    df.fillna(0.0, inplace=True)
            
    X = df[features].values
    y = df['match_goal_difference'].values
    
    # Dynamically extract Principal Components until 85% variance is kept
    pca = PCA(n_components=0.85, svd_solver='full')
    X_pca = pca.fit_transform(X)
    
    # Train unbiased Multiple Linear Regression upon exactly orthogonal space
    mlr = LinearRegression()
    mlr.fit(X_pca, y)
    
    # Finalize weighting structure: MLR_weights @ PCA_vectors
    final_weights = mlr.coef_ @ pca.components_
    
    # Resolve to strictly normalized ratio vector
    abs_weights = np.abs(final_weights)
    sum_weights = np.sum(abs_weights)
    
    if sum_weights == 0:
        w_norm = np.array([1.0 / len(features)] * len(features))
    else:
        w_norm = abs_weights / sum_weights
        
    cumulative_variance = float(np.sum(pca.explained_variance_ratio_) * 100)
    n_components = int(pca.n_components_)
    r2_score = float(mlr.score(X_pca, y))
    
    weights = {
        'w_coh': float(w_norm[0]),
        'w_txt': float(w_norm[1]),
        'w_bxt': float(w_norm[2]),
        'w_dec': float(w_norm[3]),
        'w_xg': float(w_norm[4]),
        'w_pacc': float(w_norm[5]),
        'w_ret': float(w_norm[6]),
        'w_itr': float(w_norm[7]),
        'w_dsi': float(w_norm[8]),
        'w_hes': float(w_norm[9]),
        'cumulative_variance': cumulative_variance,
        'n_components': n_components,
        'r2_score': r2_score
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(weights, f, indent=4)
        
    return weights

def train_tes_xgboost_weights(save_path, xgb_model_path):
    """
    Trains an XGBoost regressor using GroupKFold (grouped by team_id) to predict match_goal_difference.
    Standardizes features by competition and season. Extracts global Feature Gain as weights.
    """
    import json
    import numpy as np
    import xgboost as xgb
    import os
    import config
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    
    train_path = os.path.join(config.DATA_DIR, "tes_train_data.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("Match-level training data not found. Please run src/scripts/build_tes_dataset.py first.")
        
    df = pd.read_csv(train_path)
    features = ['Coh', 'TxT', 'BxT', 'Dec', 'xG', 'PAcc', 'Ret', 'ITr', 'DSI', 'HES']
    
    # Standardize these features within each competition and season
    df[features] = df.groupby(['competition_id', 'season_id'])[features].transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))
    df.fillna(0.0, inplace=True)

    X = df[features].values
    y = df['match_goal_difference'].values
    groups = df['team_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    # Heavy regularization as specified
    model = xgb.XGBRegressor(
        n_estimators=300, 
        max_depth=5, 
        learning_rate=0.05, 
        random_state=42
    )
    
    r2_scores = []
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        r2_scores.append(model.score(X_test, y_test))
        
    # Fit on all data for final weights
    model.fit(X, y)
    
    # Save the model
    os.makedirs(os.path.dirname(xgb_model_path), exist_ok=True)
    model.save_model(xgb_model_path)
    
    # Extract Feature Importance (Gain)
    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    
    weights_arr = np.zeros(len(features))
    for i, f in enumerate(features):
        f_name = f"f{i}" if f"f{i}" in importance else f
        weights_arr[i] = importance.get(f_name, importance.get(f, 0.0))
        
    sum_w = np.sum(weights_arr)
    if sum_w > 0:
        w_norm = weights_arr / sum_w
    else:
        w_norm = np.ones(len(features)) / len(features)
        
    weights = {
        'w_coh': float(w_norm[0]),
        'w_txt': float(w_norm[1]),
        'w_bxt': float(w_norm[2]),
        'w_dec': float(w_norm[3]),
        'w_xg': float(w_norm[4]),
        'w_pacc': float(w_norm[5]),
        'w_ret': float(w_norm[6]),
        'w_itr': float(w_norm[7]),
        'w_dsi': float(w_norm[8]),
        'w_hes': float(w_norm[9]),
        'r2_avg_cv': float(np.mean(r2_scores)),
        'model_architecture': 'xgboost_gain',
        'saved_model_path': xgb_model_path
    }
    
    with open(save_path, 'w') as f:
        json.dump(weights, f, indent=4)
        
    return weights


def train_tes_notebook(model_save_path):
    """
    Exact replication of TES Final Training.ipynb.
    Pipeline: RobustScaler + GradientBoostingRegressor.
    Target: match_goal_ratio. Groups: match_id. 8 differential features.
    """
    import os, joblib, numpy as np, pandas as pd, config
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    train_path = os.path.join(config.DATA_DIR, "tes_train_data.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "Match-level training data not found. "
            "Please run src/scripts/build_tes_dataset.py first."
        )

    df = pd.read_csv(train_path)
    core_metrics = ["xG", "Ret", "DSI", "BxT", "Dec"]
    df = df.sort_values(["match_id", "team_id"]).copy()
    df["team_order"] = df.groupby("match_id").cumcount()

    df_0 = df[df["team_order"] == 0].copy()
    df_1 = df[df["team_order"] == 1].copy()
    valid_match_ids = set(df_0["match_id"]).intersection(set(df_1["match_id"]))
    df_0 = df_0[df_0["match_id"].isin(valid_match_ids)]
    df_1 = df_1[df_1["match_id"].isin(valid_match_ids)]

    opp_map = {col: f"opp_{col}" for col in core_metrics}
    m0 = pd.merge(df_0, df_1[["match_id"] + core_metrics].rename(columns=opp_map), on="match_id")
    m1 = pd.merge(df_1, df_0[["match_id"] + core_metrics].rename(columns=opp_map), on="match_id")
    df_enriched = pd.concat([m0, m1], ignore_index=True)

    for col in core_metrics:
        df_enriched[f"{col}_diff"] = df_enriched[col] - df_enriched[f"opp_{col}"]

    selected_cols = ["xG_diff", "xG", "opp_xG", "Ret_diff", "opp_DSI", "BxT_diff", "Dec_diff", "Ret"]
    df_enriched = df_enriched.dropna(subset=selected_cols + ["match_goal_ratio"])

    X      = df_enriched[selected_cols]
    y      = df_enriched["match_goal_ratio"]
    groups = df_enriched["match_id"]

    gkf = GroupKFold(n_splits=5)
    all_preds, all_actuals = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        fold_model = Pipeline([
            ("scaler",    RobustScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.07, max_depth=4, random_state=42
            ))
        ])
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        all_preds.extend(fold_model.predict(X.iloc[test_idx]))
        all_actuals.extend(y.iloc[test_idx])

    all_preds, all_actuals = np.array(all_preds), np.array(all_actuals)
    rmse = float(np.sqrt(mean_squared_error(all_actuals, all_preds)))
    mae  = float(mean_absolute_error(all_actuals, all_preds))
    r2   = float(r2_score(all_actuals, all_preds))

    final_model = Pipeline([
        ("scaler",    RobustScaler()),
        ("regressor", GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.07, max_depth=4, random_state=42
        ))
    ])
    final_model.fit(X, y)

    model_data = {"model": final_model, "features": selected_cols}
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model_data, model_save_path)

    return {
        "model_type": "GradientBoostingRegressor + RobustScaler",
        "target":     "match_goal_ratio",
        "features":   selected_cols,
        "n_samples":  int(len(df_enriched)),
        "cv_rmse":    rmse,
        "cv_mae":     mae,
        "cv_r2":      r2,
        "model_path": model_save_path
    }


def predict_tes_for_team(team_dna, league_dna_df, model_save_path):
    """
    Predicts TES (match_goal_ratio) for one team using the notebook-trained model.
    Uses league-average DNA as the neutral-opponent proxy for differential features.
    Returns float clipped at 0.
    """
    import os, joblib, pandas as pd

    if not os.path.exists(model_save_path):
        return 0.0
    try:
        md    = joblib.load(model_save_path)
        model = md["model"]
        feats = md["features"]
    except Exception:
        return 0.0

    # Full alias map: model key → all known dict key formats
    # (leaderboard uses short keys; DNA profile JSON uses avg_* long keys)
    _ALIASES = {
        "xG":  ["xG", "xg", "avg_xg"],
        "Ret":  ["Ret", "ret", "Retention", "retention", "avg_ret", "avg_retention"],
        "DSI":  ["DSI", "dsi", "avg_dsi"],
        "BxT":  ["BxT", "bxt", "Basic_xT", "basic_xt", "avg_bxt", "avg_xt"],
        "Dec":  ["Dec", "dec", "Centralization", "centralization",
                 "avg_dec", "avg_centralization"],
    }

    def get_val(d, key):
        for k in _ALIASES.get(key, [key, key.lower(), f"avg_{key.lower()}"]):
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    pass
        return 0.0

    tv = {c: get_val(team_dna, c) for c in ["xG", "Ret", "DSI", "BxT", "Dec"]}
    ov = {}
    for c in ["xG", "Ret", "DSI", "BxT", "Dec"]:
        try:
            col_data = league_dna_df[c] if c in league_dna_df.columns else None
            ov[c] = float(col_data.mean()) if col_data is not None and not col_data.empty else tv[c]
        except Exception:
            ov[c] = tv[c]

    fv = {
        "xG_diff":  tv["xG"]  - ov["xG"],
        "xG":       tv["xG"],
        "opp_xG":   ov["xG"],
        "Ret_diff": tv["Ret"] - ov["Ret"],
        "opp_DSI":  ov["DSI"],
        "BxT_diff": tv["BxT"] - ov["BxT"],
        "Dec_diff": tv["Dec"] - ov["Dec"],
        "Ret":      tv["Ret"],
    }
    try:
        pred = float(model.predict(pd.DataFrame([fv])[feats])[0])
    except Exception:
        return 0.0
    return max(0.0, pred)

def get_tes_weights(save_path):
    """
    Loads TES weights from JSON, returns heuristic defaults if not found.
    """
    import os
    import json
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                w = json.load(f)
                return (
                    w.get('w_coh', 0.125), w.get('w_txt', 0.125), w.get('w_bxt', 0.125),
                    w.get('w_dec', 0.1), w.get('w_xg', 0.1), w.get('w_pacc', 0.1),
                    w.get('w_ret', 0.1), w.get('w_itr', 0.1), w.get('w_dsi', 0.1), w.get('w_hes', 0.1)
                )
        except Exception:
            pass
    return 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

def calculate_championship_leaderboard(matches_df, comp_name, season_name, dna_dir, xt_model=None, trans_checkpoint_path=None, engine_type='Hybrid PCA-MLR'):
    """
    Constructs a ranking dataframe combining match results and DNA metrics 
    for all teams in the given Competition & Season.
    """
    import os
    import json
    import pandas as pd
    from utils.data_loader import load_statsbomb_data, preprocess_passes
    from engine.xt_model import apply_xt_to_passes
    from engine.transgoalnet import apply_transgoalnet_inference
    
    # 1. Calculate Standings (W/L Spread)
    standings_df = get_team_match_results(matches_df)
    if standings_df.empty:
        return standings_df
        
    safe_comp = comp_name.replace("/", "_").replace(" ", "_")
    safe_season = season_name.replace("/", "_").replace(" ", "_")
    
    # We will collect DNA metrics for teams that have them saved.
    dna_records = []
    
    for team in standings_df['Team']:
        safe_team = team.replace("/", "_").replace(" ", "_")
        profile_path = os.path.join(dna_dir, safe_comp, safe_season, safe_team, "dna_profile.json")
        
        coh = 0.0
        txt = 0.0
        bxt = 0.0
        cent = 0.0
        xg = 0.0
        pacc = 0.0
        ret = 0.0
        itr = 0.0
        dsi = 0.0
        hes = 0.0
        has_dna = False
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)
                    overall = data.get("overall", {})
                    if overall:
                        has_dna = True
                        coh = overall.get("avg_cohesion", 0.0)
                        txt = overall.get("avg_trans_xt", 0.0)
                        bxt = overall.get("avg_xt", 0.0)
                        cent = overall.get("avg_centralization", 0.0)
                        xg = overall.get("avg_xg", 0.0)
                        pacc = overall.get("avg_pass_acc", 0.0)
                        ret = overall.get("avg_retention", 0.0)
                        itr = overall.get("avg_itrans", 0.0)
                        dsi = overall.get("avg_dsi", 0.0)
                        hes = overall.get("avg_hes", 0.0)
            except Exception:
                pass
                
        if not has_dna:
            # Dynamically compute basic dna values permanently to prevent empty tables in future
            try:
                team_matches = matches_df[(matches_df['home_team'] == team) | (matches_df['away_team'] == team)]
                team_raw_df = load_statsbomb_data(team_matches, team, limit_matches=None, filter_team=False)
                
                if not team_raw_df.empty:
                    my_team_df = team_raw_df[team_raw_df['team'] == team].copy()
                    pass_df = preprocess_passes(my_team_df)
                    
                    if xt_model is not None:
                        pass_df = apply_xt_to_passes(pass_df, xt_model=xt_model)
                        
                    if trans_checkpoint_path is not None and xt_model is not None:
                        try:
                            pass_df, _ = apply_transgoalnet_inference(pass_df, basic_xt_model=xt_model, model_checkpoint_path=trans_checkpoint_path)
                        except Exception:
                            pass
                        
                    dna_comprehensive = generate_and_save_comprehensive_dna(pass_df, team_matches, team, comp_name, season_name, dna_dir, team_raw_df)
                    overall = dna_comprehensive.get("overall", {})
                    
                    has_dna = True
                    coh = overall.get("avg_cohesion", 0.0)
                    txt = overall.get("avg_trans_xt", 0.0)
                    bxt = overall.get("avg_xt", 0.0)
                    cent = overall.get("avg_centralization", 0.0)
                    xg = overall.get("avg_xg", 0.0)
                    pacc = overall.get("avg_pass_acc", 0.0)
                    ret = overall.get("avg_retention", 0.0)
                    itr = overall.get("avg_itrans", 0.0)
                    dsi = overall.get("avg_dsi", 0.0)
                    hes = overall.get("avg_hes", 0.0)
            except Exception:
                pass
                
        dna_records.append({
            'Team': team,
            'Has_DNA': has_dna,
            'Cohesion': coh,
            'Trans_xT': txt,
            'Basic_xT': bxt,
            'Centralization': cent,
            'xG': xg,
            'Pass_Acc': pacc,
            'Retention': ret,
            'ITrans': itr,
            'DSI': dsi,
            'HES': hes
        })
        
    dna_df = pd.DataFrame(dna_records)
    
    # Merge
    merged_df = pd.merge(standings_df, dna_df, on="Team")
    
    # Calculate Min-Max for DNA metrics among teams that HAVE DNA
    valid_dna = merged_df[merged_df['Has_DNA'] == True]
    
    if not valid_dna.empty and len(valid_dna) > 1:
        def min_max(col):
            c_min = valid_dna[col].min()
            c_max = valid_dna[col].max()
            if c_max == c_min: return pd.Series([0.5]*len(valid_dna), index=valid_dna.index)
            return (valid_dna[col] - c_min) / (c_max - c_min)
            
        merged_df.loc[valid_dna.index, 'Coh_Norm'] = min_max('Cohesion')
        merged_df.loc[valid_dna.index, 'TxT_Norm'] = min_max('Trans_xT')
        merged_df.loc[valid_dna.index, 'BxT_Norm'] = min_max('Basic_xT')
        merged_df.loc[valid_dna.index, 'Dec_Norm'] = 1.0 - min_max('Centralization')
        merged_df.loc[valid_dna.index, 'xG_Norm'] = min_max('xG')
        merged_df.loc[valid_dna.index, 'PAcc_Norm'] = min_max('Pass_Acc')
        merged_df.loc[valid_dna.index, 'Ret_Norm'] = 1.0 - min_max('Retention') # Or regular if you want
        merged_df.loc[valid_dna.index, 'ITr_Norm'] = min_max('ITrans')
        merged_df.loc[valid_dna.index, 'DSI_Norm'] = min_max('DSI')
        merged_df.loc[valid_dna.index, 'HES_Norm'] = min_max('HES')
        
    else:
        merged_df['Coh_Norm'] = 0.5
        merged_df['TxT_Norm'] = 0.5
        merged_df['BxT_Norm'] = 0.5
        merged_df['Dec_Norm'] = 0.5
        merged_df['xG_Norm'] = 0.5
        merged_df['PAcc_Norm'] = 0.5
        merged_df['Ret_Norm'] = 0.5
        merged_df['ITr_Norm'] = 0.5
        merged_df['DSI_Norm'] = 0.5
        merged_df['HES_Norm'] = 0.5
        
    # Calculate TES using the notebook-trained model (predict_tes_for_team)
    import config
    # Build a minimal league-average DataFrame from teams that have DNA
    _league_cols = ['xG', 'Ret', 'DSI', 'BxT', 'Dec']
    _league_df = merged_df[merged_df['Has_DNA'] == True][['Basic_xT', 'Retention', 'DSI', 'HES', 'Centralization', 'xG', 'Pass_Acc']].rename(
        columns={'Basic_xT': 'BxT', 'Retention': 'Ret', 'Centralization': 'Dec'}
    ) if not merged_df.empty else pd.DataFrame()
    # Ensure we have all required cols
    for _c in _league_cols:
        if _c not in _league_df.columns:
            _league_df[_c] = 0.0
    _league_df = _league_df[_league_cols]

    def _make_team_dna_dict(row):
        return {
            'xG': row.get('xG', 0.0),
            'Ret': row.get('Retention', 0.0),
            'DSI': row.get('DSI', 0.0),
            'BxT': row.get('Basic_xT', 0.0),
            'Dec': row.get('Centralization', 0.0),
        }

    merged_df['TES'] = merged_df.apply(
        lambda row: predict_tes_for_team(
            _make_team_dna_dict(row), _league_df, config.TES_MODEL_JOBLIB
        ) if row.get('Has_DNA', False) else 0.0,
        axis=1
    )

    # Normalise TES to 0-100 so the leaderboard table and radar ring
    # always show the same number.
    merged_df.loc[merged_df['Has_DNA'] == False, 'TES'] = 0.0
    _has_mask = merged_df['Has_DNA'] == True
    _t_min, _t_max = 0.0, 1.0   # fallback bounds
    if _has_mask.sum() > 1:
        _t_min = merged_df.loc[_has_mask, 'TES'].min()
        _t_max = merged_df.loc[_has_mask, 'TES'].max()
        if _t_max > _t_min:
            merged_df.loc[_has_mask, 'TES'] = (
                (merged_df.loc[_has_mask, 'TES'] - _t_min) / (_t_max - _t_min) * 100
            )
    # Store raw bounds so radar TES scoring can replicate identical normalisation
    merged_df['TES_raw_min'] = _t_min
    merged_df['TES_raw_max'] = _t_max

    # Sort by TES
    merged_df = merged_df.sort_values('TES', ascending=False).reset_index(drop=True)

    return merged_df

def calculate_all_time_leaderboard(comp_name, comp_id, get_matches_func, get_competitions_func, dna_dir, save_dir, xt_model=None, trans_checkpoint_path=None, year_threshold=None, engine_type='Hybrid PCA-MLR', force_refresh=False):
    """
    Calculates the 'All-Time' Championship DNA Leaderboard for a selected competition.
    It aggregates all available matches and DNA profiles across all seasons.
    Caches the result to a CSV file to load instantly on future requests.
    """
    import os
    import json
    import pandas as pd
    from utils.data_loader import load_statsbomb_data, preprocess_passes
    from engine.xt_model import apply_xt_to_passes
    from engine.transgoalnet import apply_transgoalnet_inference
    
    safe_comp = comp_name.replace("/", "_").replace(" ", "_")
    
    # Save inside dna_dir / safe_comp
    comp_dna_dir = os.path.join(dna_dir, safe_comp)
    os.makedirs(comp_dna_dir, exist_ok=True)
    
    thresh_str = f"_since_{year_threshold}" if year_threshold is not None else ""
    save_path = os.path.join(comp_dna_dir, f"all_time_leaderboard{thresh_str}.csv")
    
    # Check cache
    if not force_refresh and os.path.exists(save_path):
        return pd.read_csv(save_path)
        
    comps = get_competitions_func()
    if comp_name == 'Global':
        comp_data = comps
    else:
        comp_data = comps[comps['competition_name'] == comp_name]
    
    if comp_data.empty:
        return pd.DataFrame()
        
    all_matches_list = []
    valid_seasons = []
    
    # 1. Fetch matches for all seasons
    for _, row in comp_data.iterrows():
        s_id = row['season_id']
        s_name = row['season_name']
        
        if year_threshold is not None:
            try:
                start_year = int(str(s_name).split('/')[0])
                if start_year < year_threshold:
                    continue
            except:
                pass
                
        c_name = row['competition_name']
        valid_seasons.append((c_name, s_name))
        
        c_id = row['competition_id']
        matches = get_matches_func(c_id, s_id)
        if not matches.empty:
            all_matches_list.append(matches)
            
    if not all_matches_list:
        return pd.DataFrame()
        
    all_matches_df = pd.concat(all_matches_list, ignore_index=True)
    
    # 2. Calculate match standings (Win/Loss Ratios over all historical matches)
    standings_df = get_team_match_results(all_matches_df)
    
    dna_records = {} # Team -> list of DNA dicts from all seasons
    
    # 3. Gather DNA profiles across all seasons
    for team in standings_df['Team']:
        safe_team = team.replace("/", "_").replace(" ", "_")
        dna_records[team] = []
        
        for c_name, s_name in valid_seasons:
            safe_c_name = c_name.replace("/", "_").replace(" ", "_")
            safe_s_name = s_name.replace("/", "_").replace(" ", "_")
            profile_path = os.path.join(dna_dir, safe_c_name, safe_s_name, safe_team, "dna_profile.json")
            
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, "r") as f:
                        data = json.load(f)
                        overall = data.get("overall", {})
                        if overall:
                            dna_records[team].append({
                                'Cohesion': overall.get("avg_cohesion", 0.0),
                                'Trans_xT': overall.get("avg_trans_xt", 0.0),
                                'Basic_xT': overall.get("avg_xt", 0.0),
                                'Centralization': overall.get("avg_centralization", 0.0),
                                'xG': overall.get("avg_xg", 0.0),
                                'Pass_Acc': overall.get("avg_pass_acc", 0.0),
                                'Retention': overall.get("avg_retention", 0.0),
                                'ITrans': overall.get("avg_itrans", 0.0),
                                'DSI': overall.get("avg_dsi", 0.0),
                                'HES': overall.get("avg_hes", 0.0)
                            })
                except Exception:
                    pass
                    
    # Initialize variables for dynamic computation
    final_dna_rows = []
    
    for team, profiles in dna_records.items():
        if profiles:
            avg_coh = np.mean([p['Cohesion'] for p in profiles])
            avg_txt = np.mean([p['Trans_xT'] for p in profiles])
            avg_bxt = np.mean([p['Basic_xT'] for p in profiles])
            avg_cent = np.mean([p['Centralization'] for p in profiles])
            avg_xg = np.mean([p['xG'] for p in profiles])
            avg_pacc = np.mean([p['Pass_Acc'] for p in profiles])
            avg_ret = np.mean([p['Retention'] for p in profiles])
            avg_itr = np.mean([p['ITrans'] for p in profiles])
            avg_dsi = np.mean([p['DSI'] for p in profiles])
            avg_hes = np.mean([p['HES'] for p in profiles])
            final_dna_rows.append({
                'Team': team,
                'Has_DNA': True,
                'Cohesion': avg_coh,
                'Trans_xT': avg_txt,
                'Basic_xT': avg_bxt,
                'Centralization': avg_cent,
                'xG': avg_xg,
                'Pass_Acc': avg_pacc,
                'Retention': avg_ret,
                'ITrans': avg_itr,
                'DSI': avg_dsi,
                'HES': avg_hes,
                'Seasons_Saved': len(profiles)
            })
        else:
            # DYNAMIC COMPUTATION: Pull raw events, securely save profile, and append traits
            try:
                team_matches = all_matches_df[(all_matches_df['home_team'] == team) | (all_matches_df['away_team'] == team)]
                team_raw_df = load_statsbomb_data(team_matches, team, limit_matches=None, filter_team=False)
                
                if not team_raw_df.empty:
                    my_team_df = team_raw_df[team_raw_df['team'] == team].copy()
                    pass_df = preprocess_passes(my_team_df)
                    
                    if xt_model is not None:
                        pass_df = apply_xt_to_passes(pass_df, xt_model=xt_model)
                        
                    if trans_checkpoint_path is not None and xt_model is not None:
                        try:
                            pass_df, _ = apply_transgoalnet_inference(pass_df, basic_xt_model=xt_model, model_checkpoint_path=trans_checkpoint_path)
                        except Exception:
                            pass
                            
                    # Use the first available season from their matches for structure:
                    t_season = team_matches['season'].iloc[0] if 'season' in team_matches.columns else "Unknown_Season"
                    dna_comprehensive = generate_and_save_comprehensive_dna(pass_df, team_matches, team, comp_name, t_season, dna_dir, team_raw_df)
                    
                    overall = dna_comprehensive.get("overall", {})
                    
                    final_dna_rows.append({
                        'Team': team,
                        'Has_DNA': True,
                        'Cohesion': overall.get("avg_cohesion", 0.0),
                        'Trans_xT': overall.get("avg_trans_xt", 0.0),
                        'Basic_xT': overall.get("avg_xt", 0.0),
                        'Centralization': overall.get("avg_centralization", 0.0),
                        'xG': overall.get("avg_xg", 0.0),
                        'Pass_Acc': overall.get("avg_pass_acc", 0.0),
                        'Retention': overall.get("avg_retention", 0.0),
                        'ITrans': overall.get("avg_itrans", 0.0),
                        'DSI': overall.get("avg_dsi", 0.0),
                        'HES': overall.get("avg_hes", 0.0),
                        'Seasons_Saved': 1
                    })
                else:
                    final_dna_rows.append({
                        'Team': team, 'Has_DNA': False, 'Cohesion': 0.0, 'Trans_xT': 0.0, 'Basic_xT': 0.0, 'Centralization': 0.0, 'xG': 0.0, 'Pass_Acc': 0.0, 'Retention': 0.0, 'ITrans': 0.0, 'DSI': 0.0, 'HES': 0.0, 'Seasons_Saved': 0
                    })
            except Exception:
                final_dna_rows.append({
                    'Team': team, 'Has_DNA': False, 'Cohesion': 0.0, 'Trans_xT': 0.0, 'Basic_xT': 0.0, 'Centralization': 0.0, 'xG': 0.0, 'Pass_Acc': 0.0, 'Retention': 0.0, 'ITrans': 0.0, 'DSI': 0.0, 'HES': 0.0, 'Seasons_Saved': 0
                })
            
    dna_df = pd.DataFrame(final_dna_rows)
    merged_df = pd.merge(standings_df, dna_df, on="Team")
    
    # 4. Normalize and calculate TES
    valid_dna = merged_df[merged_df['Has_DNA'] == True]
    if not valid_dna.empty and len(valid_dna) > 1:
        def min_max(col):
            c_min = valid_dna[col].min()
            c_max = valid_dna[col].max()
            if c_max == c_min: return pd.Series([0.5]*len(valid_dna), index=valid_dna.index)
            return (valid_dna[col] - c_min) / (c_max - c_min)
            
        merged_df.loc[valid_dna.index, 'Coh_Norm'] = min_max('Cohesion')
        merged_df.loc[valid_dna.index, 'TxT_Norm'] = min_max('Trans_xT')
        merged_df.loc[valid_dna.index, 'BxT_Norm'] = min_max('Basic_xT')
        merged_df.loc[valid_dna.index, 'Dec_Norm'] = 1.0 - min_max('Centralization')
        merged_df.loc[valid_dna.index, 'xG_Norm'] = min_max('xG')
        merged_df.loc[valid_dna.index, 'PAcc_Norm'] = min_max('Pass_Acc')
        merged_df.loc[valid_dna.index, 'Ret_Norm'] = min_max('Retention')
        merged_df.loc[valid_dna.index, 'ITr_Norm'] = min_max('ITrans')
        merged_df.loc[valid_dna.index, 'DSI_Norm'] = min_max('DSI')
        merged_df.loc[valid_dna.index, 'HES_Norm'] = min_max('HES')
    else:
        merged_df['Coh_Norm'] = 0.5
        merged_df['TxT_Norm'] = 0.5
        merged_df['BxT_Norm'] = 0.5
        merged_df['Dec_Norm'] = 0.5
        merged_df['xG_Norm'] = 0.5
        merged_df['PAcc_Norm'] = 0.5
        merged_df['Ret_Norm'] = 0.5
        merged_df['ITr_Norm'] = 0.5
        merged_df['DSI_Norm'] = 0.5
        merged_df['HES_Norm'] = 0.5
        
    # Calculate TES using the notebook-trained model
    import config
    _league_cols = ['xG', 'Ret', 'DSI', 'BxT', 'Dec']
    _league_df = merged_df[merged_df['Has_DNA'] == True][['Basic_xT', 'Retention', 'DSI', 'HES', 'Centralization', 'xG', 'Pass_Acc']].rename(
        columns={'Basic_xT': 'BxT', 'Retention': 'Ret', 'Centralization': 'Dec'}
    ) if not merged_df.empty else pd.DataFrame()
    for _c in _league_cols:
        if _c not in _league_df.columns:
            _league_df[_c] = 0.0
    _league_df = _league_df[_league_cols]

    def _make_team_dna_dict(row):
        return {
            'xG': row.get('xG', 0.0),
            'Ret': row.get('Retention', 0.0),
            'DSI': row.get('DSI', 0.0),
            'BxT': row.get('Basic_xT', 0.0),
            'Dec': row.get('Centralization', 0.0),
        }

    merged_df['TES'] = merged_df.apply(
        lambda row: predict_tes_for_team(
            _make_team_dna_dict(row), _league_df, config.TES_MODEL_JOBLIB
        ) if row.get('Has_DNA', False) else 0.0,
        axis=1
    )

    # Normalise TES to 0-100 (same scale as radar ring)
    merged_df.loc[merged_df['Has_DNA'] == False, 'TES'] = 0.0
    _has_mask = merged_df['Has_DNA'] == True
    _t_min, _t_max = 0.0, 1.0
    if _has_mask.sum() > 1:
        _t_min = merged_df.loc[_has_mask, 'TES'].min()
        _t_max = merged_df.loc[_has_mask, 'TES'].max()
        if _t_max > _t_min:
            merged_df.loc[_has_mask, 'TES'] = (
                (merged_df.loc[_has_mask, 'TES'] - _t_min) / (_t_max - _t_min) * 100
            )
    merged_df['TES_raw_min'] = _t_min
    merged_df['TES_raw_max'] = _t_max

    merged_df = merged_df.sort_values('TES', ascending=False).reset_index(drop=True)

    # 5. Save and return
    merged_df.to_csv(save_path, index=False)

    return merged_df

