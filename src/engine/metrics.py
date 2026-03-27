import networkx as nx
import numpy as np

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

def calculate_team_dna(df):
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
    
    for match_id in match_ids:
        # If 'match_id' isn't in df, just use the whole df
        m_df = df[df['match_id'] == match_id] if 'match_id' in df.columns else df
        
        volumes.append(len(m_df))
        c, co, a = get_network_metrics(m_df)
        centralizations.append(c)
        cohesions.append(co)
        conns.append(a)
        
        m_xt = m_df['xT'].sum() if 'xT' in m_df.columns else 0.0
        xts.append(m_xt)
        
        m_trans_xt = m_df['Trans_xT'].sum() if 'Trans_xT' in m_df.columns else 0.0
        trans_xts.append(m_trans_xt)
        
    avg_volume = np.mean(volumes) if volumes else 0.0
    avg_cent = np.mean(centralizations) if centralizations else 0.0
    avg_coh = np.mean(cohesions) if cohesions else 0.0
    avg_conns = np.mean(conns) if conns else 0.0
    avg_xt = np.mean(xts) if xts else 0.0
    avg_trans_xt = np.mean(trans_xts) if trans_xts else 0.0
    
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
        "top_threat_creators": top_creators,
        "top_trans_threat_creators": top_trans_creators
    }
    
    return dna_metrics

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
