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
    Calculates the Team DNA metrics from the full passes dataframe.
    """
    if df.empty:
        return {}
    
    # 1. Volume
    volume = len(df)
    
    # 2. Centralization & Cohesion & Active Connections
    cent, coh, active_conns = get_network_metrics(df)
    
    # 3. xT Metrics
    total_xt = df['xT'].sum() if 'xT' in df.columns else 0.0
    xt_per_pass = total_xt / volume if volume > 0 else 0.0
    
    # 4. Top Threat Creators
    top_creators = {}
    if 'xT' in df.columns and 'player_name' in df.columns:
        player_xt = df.groupby('player_name')['xT'].sum().sort_values(ascending=False)
        top_creators = player_xt.head(3).to_dict()
        
    dna_metrics = {
        "pass_volume": volume,
        "active_connections": active_conns,
        "centralization": float(cent),
        "cohesion": float(coh),
        "total_xt": float(total_xt),
        "xt_per_pass": float(xt_per_pass),
        "top_threat_creators": top_creators
    }
    
    return dna_metrics
