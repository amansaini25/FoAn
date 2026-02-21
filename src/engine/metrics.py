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
