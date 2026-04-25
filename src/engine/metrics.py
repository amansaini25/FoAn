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
            m_xg = m_raw['shot_statsbomb_xg'].sum() if 'shot_statsbomb_xg' in m_raw.columns else 0.0
            xgs.append(m_xg)
            
            passes = m_raw[m_raw['type'] == 'Pass']
            total_passes = len(passes)
            # m_df only contains successful passes for this team
            pass_acc = len(m_df) / total_passes if total_passes > 0 else 0.0
            pass_accs.append(pass_acc)
            
            losses = len(passes[passes['pass_outcome'].notnull()]) + len(m_raw[m_raw['type'].isin(['Dispossessed', 'Miscontrol'])])
            m_poss_count = m_raw['possession'].nunique() if 'possession' in m_raw.columns else 1.0
            retention = losses / m_poss_count if m_poss_count > 0 else 0.0
            retentions.append(retention)
            
            m_poss_duration = m_raw['duration'].sum() if 'duration' in m_raw.columns else 0.0
            itrans = m_trans_xt / m_poss_duration if m_poss_duration > 0 else 0.0
            itrans_list.append(itrans)
        else:
            xgs.append(0.0)
            pass_accs.append(0.0)
            retentions.append(0.0)
            itrans_list.append(0.0)
        
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
    comp_df['time_bin'] = pd.cut(comp_df['minute'], bins=bins, labels=labels, right=False)
    
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

def train_tes_pca_weights(leaderboard_df, save_path):
    """
    Trains a Hybrid PCA-MLR model from the leaderboard features
    to extract variance dynamically and bind it onto Win_Ratio objectively.
    """
    import json
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    train_df = leaderboard_df[leaderboard_df['Has_DNA'] == True].copy()
    
    if len(train_df) < 5:
        raise ValueError("Not enough teams with DNA profiles to train PCA reliably.")
        
    features = ['Coh_Norm', 'TxT_Norm', 'BxT_Norm', 'Dec_Norm', 'xG_Norm', 'PAcc_Norm', 'Ret_Norm', 'ITr_Norm']
    
    for f in features:
        if f not in train_df.columns:
            train_df[f] = 0.5
            
    X = train_df[features].values
    y = train_df['Win_Ratio'].values
    
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
        'cumulative_variance': cumulative_variance,
        'n_components': n_components,
        'r2_score': r2_score
    }
    
    with open(save_path, 'w') as f:
        json.dump(weights, f, indent=4)
        
    return weights

def train_tes_xgboost_weights(leaderboard_df, save_path, xgb_model_path):
    """
    Trains an XGBoost model tracking normalized feature spaces mathematically
    and extracts universally agnostic variable SHAP weights.
    """
    import json
    import numpy as np
    import xgboost as xgb
    import shap
    from sklearn.preprocessing import StandardScaler
    
    train_df = leaderboard_df[leaderboard_df['Has_DNA'] == True].copy()
    
    if len(train_df) < 5:
        raise ValueError("Not enough teams with DNA profiles to train XGBoost reliably.")
        
    features = ['Coh_Norm', 'TxT_Norm', 'BxT_Norm', 'Dec_Norm', 'xG_Norm', 'PAcc_Norm', 'Ret_Norm', 'ITr_Norm']
    
    for f in features:
        if f not in train_df.columns:
            train_df[f] = 0.5
            
    X = train_df[features].values
    y = train_df['Win_Ratio'].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    # Train XGBRegressor mapped iteratively
    model = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, random_state=42, early_stopping_rounds=20)
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    # Dump tree logic to the C++ assets folder
    import os
    os.makedirs(os.path.dirname(xgb_model_path), exist_ok=True)
    model.save_model(xgb_model_path)
    
    # Calculate SHAP global importances
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Absolute Mean magnitude represents discrete global feature alignment
    global_importance = np.mean(np.abs(shap_values), axis=0)
    
    sum_weights = np.sum(global_importance)
    if sum_weights > 0:
        w_norm = global_importance / sum_weights
    else:
        w_norm = np.array([1.0 / len(features)] * len(features))
        
    r2_train = float(model.score(X_train_scaled, y_train))
    r2_test = float(model.score(X_test_scaled, y_test))
    
    weights = {
        'w_coh': float(w_norm[0]),
        'w_txt': float(w_norm[1]),
        'w_bxt': float(w_norm[2]),
        'w_dec': float(w_norm[3]),
        'w_xg': float(w_norm[4]),
        'w_pacc': float(w_norm[5]),
        'w_ret': float(w_norm[6]),
        'w_itr': float(w_norm[7]),
        'r2_train': r2_train,
        'r2_test': r2_test,
        'model_architecture': 'xgboost',
        'saved_model_path': xgb_model_path
    }
    
    with open(save_path, 'w') as f:
        json.dump(weights, f, indent=4)
        
    return weights

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
                    w.get('w_dec', 0.125), w.get('w_xg', 0.125), w.get('w_pacc', 0.125),
                    w.get('w_ret', 0.125), w.get('w_itr', 0.125)
                )
        except Exception:
            pass
    return 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125

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
                        
                    dna_comprehensive = generate_and_save_comprehensive_dna(pass_df, team_matches, team, comp_name, season_name, dna_dir, my_team_df)
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
            'ITrans': itr
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
        merged_df.loc[valid_dna.index, 'Ret_Norm'] = 1.0 - min_max('Retention')
        merged_df.loc[valid_dna.index, 'ITr_Norm'] = min_max('ITrans')
        
    else:
        merged_df['Coh_Norm'] = 0.5
        merged_df['TxT_Norm'] = 0.5
        merged_df['BxT_Norm'] = 0.5
        merged_df['Dec_Norm'] = 0.5
        merged_df['xG_Norm'] = 0.5
        merged_df['PAcc_Norm'] = 0.5
        merged_df['Ret_Norm'] = 0.5
        merged_df['ITr_Norm'] = 0.5
        
    # Calculate TES
    import config
    # Dynamically fetch global historically trained weightings
    if engine_type == 'Hybrid PCA-MLR':
        weights_path = os.path.join(config.ASSETS_DIR, "tes_pca_weights.json")
    else:
        weights_path = os.path.join(config.ASSETS_DIR, "tes_xgboost_weights.json")
    w_coh, w_txt, w_bxt, w_dec, w_xg, w_pacc, w_ret, w_itr = get_tes_weights(weights_path)
    
    merged_df['TES'] = (w_coh * merged_df['Coh_Norm']) + \
                       (w_txt * merged_df['TxT_Norm']) + \
                       (w_bxt * merged_df['BxT_Norm']) + \
                       (w_dec * merged_df['Dec_Norm']) + \
                       (w_xg * merged_df['xG_Norm']) + \
                       (w_pacc * merged_df['PAcc_Norm']) + \
                       (w_ret * merged_df['Ret_Norm']) + \
                       (w_itr * merged_df['ITr_Norm'])
                       
    # If a team has no DNA saved, set TES to 0
    merged_df.loc[merged_df['Has_DNA'] == False, 'TES'] = 0.0
    
    # Sort by TES
    merged_df = merged_df.sort_values('TES', ascending=False).reset_index(drop=True)
    
    return merged_df

def calculate_all_time_leaderboard(comp_name, comp_id, get_matches_func, get_competitions_func, dna_dir, save_dir, xt_model=None, trans_checkpoint_path=None, year_threshold=None, engine_type='Hybrid PCA-MLR'):
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
    thresh_str = f"_since_{year_threshold}" if year_threshold is not None else ""
    save_path = os.path.join(save_dir, f"{safe_comp}_all_seasons{thresh_str}.csv")
    
    # Check cache
    if os.path.exists(save_path):
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
                                'ITrans': overall.get("avg_itrans", 0.0)
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
                            
                    # Since this aggregates multiple seasons, generate it using the target comp & first detected mismatching season (or aggregate format).
                    # Actually, if we just want it to save, we use the first available season from their matches for structure:
                    t_season = team_matches['season'].iloc[0] if 'season' in team_matches.columns else "Unknown_Season"
                    dna_comprehensive = generate_and_save_comprehensive_dna(pass_df, team_matches, team, comp_name, t_season, dna_dir, my_team_df)
                    
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
                        'Seasons_Saved': 1
                    })
                else:
                    final_dna_rows.append({
                        'Team': team, 'Has_DNA': False, 'Cohesion': 0.0, 'Trans_xT': 0.0, 'Basic_xT': 0.0, 'Centralization': 0.0, 'xG': 0.0, 'Pass_Acc': 0.0, 'Retention': 0.0, 'ITrans': 0.0, 'Seasons_Saved': 0
                    })
            except Exception:
                final_dna_rows.append({
                    'Team': team, 'Has_DNA': False, 'Cohesion': 0.0, 'Trans_xT': 0.0, 'Basic_xT': 0.0, 'Centralization': 0.0, 'xG': 0.0, 'Pass_Acc': 0.0, 'Retention': 0.0, 'ITrans': 0.0, 'Seasons_Saved': 0
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
    else:
        merged_df['Coh_Norm'] = 0.5
        merged_df['TxT_Norm'] = 0.5
        merged_df['BxT_Norm'] = 0.5
        merged_df['Dec_Norm'] = 0.5
        merged_df['xG_Norm'] = 0.5
        merged_df['PAcc_Norm'] = 0.5
        merged_df['Ret_Norm'] = 0.5
        merged_df['ITr_Norm'] = 0.5
        
    import config
    if engine_type == 'Hybrid PCA-MLR':
        weights_path = os.path.join(config.ASSETS_DIR, "tes_pca_weights.json")
    else:
        weights_path = os.path.join(config.ASSETS_DIR, "tes_xgboost_weights.json")
    w_coh, w_txt, w_bxt, w_dec, w_xg, w_pacc, w_ret, w_itr = get_tes_weights(weights_path)

    merged_df['TES'] = (w_coh * merged_df['Coh_Norm']) + \
                       (w_txt * merged_df['TxT_Norm']) + \
                       (w_bxt * merged_df['BxT_Norm']) + \
                       (w_dec * merged_df['Dec_Norm']) + \
                       (w_xg * merged_df['xG_Norm']) + \
                       (w_pacc * merged_df['PAcc_Norm']) + \
                       (w_ret * merged_df['Ret_Norm']) + \
                       (w_itr * merged_df['ITr_Norm'])
                       
    merged_df.loc[merged_df['Has_DNA'] == False, 'TES'] = 0.0
    merged_df = merged_df.sort_values('TES', ascending=False).reset_index(drop=True)
    
    # 5. Save and return
    merged_df.to_csv(save_path, index=False)
    
    return merged_df

