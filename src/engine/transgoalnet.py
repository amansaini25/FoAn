import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from engine.xt_model import ExpectedThreat

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x, rel_bias):
        B, N, _ = x.shape
        resid = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if rel_bias is not None:
            scores = scores + rel_bias.permute(0, 3, 1, 2)
            
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        out = self.out_proj(out)
        
        x = resid + out
        resid = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = resid + x
        return x

class TransGoalNet(nn.Module):
    def __init__(self, node_dim=4, edge_dim=5, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, nodes, edge_attrs):
        # nodes: (B, N, node_dim)
        # edge_attrs: (B, N, N, edge_dim)
        B, N, _ = nodes.shape
        x = self.node_proj(nodes)
        
        rel_bias = None
        if edge_attrs is not None:
            rel_bias = self.edge_proj(edge_attrs) # (B, N, N, num_heads)
            
        for layer in self.layers:
            x = layer(x, rel_bias)
            
        # Global mean pooling
        z = torch.mean(x, dim=1)
        
        # Predict xT change
        out = F.relu(self.fc1(z))
        y_hat = self.fc2(out)
        
        return y_hat, x

def prepare_transgoalnet_dataset(actions_df, basic_xt_model):
    """ Builds Graph nodes and edges for batches. """
    actions_df = actions_df.copy()
    actions_df['xT'] = basic_xt_model.rate(actions_df).fillna(0.0)
    
    # We will build dataset per match to establish players
    graphs = []
    
    for match_id, match_df in actions_df.groupby('match_id'):
        players = match_df['player_name'].unique().tolist()
        player_to_idx = {p: i for i, p in enumerate(players)}
        N = len(players)
        
        if N == 0: continue
        
        # Simple node features: count of actions, moves, successes
        node_feats = np.zeros((N, 4))
        for p, idx in player_to_idx.items():
            p_df = match_df[match_df['player_name'] == p]
            node_feats[idx, 0] = len(p_df) # Total involvements
            node_feats[idx, 1] = len(p_df[p_df['type'] == 'move']) # Passes/Carries
            node_feats[idx, 2] = len(p_df[p_df['result'] == 'success'])
            node_feats[idx, 3] = p_df['xT'].mean() if not p_df.empty else 0.0
            
        # Max N we handle: zero padding to max players across matches (assume 30)
        MAX_N = 30
        padded_nodes = np.zeros((MAX_N, 4))
        
        valid_N = min(N, MAX_N)
        padded_nodes[:valid_N, :] = node_feats[:valid_N, :]
        
        # Now every action is a graph instance!
        for i, row in match_df.iterrows():
            if pd.isna(row['player_name']) or row['player_name'] not in player_to_idx:
                continue
            idx = player_to_idx[row['player_name']]
            if idx >= MAX_N: continue
            
            # Edge features bias (MAX_N x MAX_N x 5)
            edge_feats = np.zeros((MAX_N, MAX_N, 5))
            # Put an "edge" from player to themselves or something representing the action
            # Since we just model one event primarily, we put it at (idx, idx) for self or (idx, prev)
            edge_feats[idx, idx, 0] = row['start_x']
            edge_feats[idx, idx, 1] = row['start_y']
            edge_feats[idx, idx, 2] = row['end_x']
            edge_feats[idx, idx, 3] = row['end_y']
            edge_feats[idx, idx, 4] = row['xT']
            
            graphs.append({
                'nodes': padded_nodes,
                'edges': edge_feats,
                'target': row['xT'],
                'player_idx': idx,
                'match_id': row['match_id'],
                'idx_orig': i # to map back
            })
            
    return graphs, MAX_N

def train_transgoalnet(graphs, max_n, epochs=15, batch_size=64, lr=1e-3, device='cuda'):
    import config
    print(f"Training TransGoalNet with {len(graphs)} samples on {device}...")
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Shuffle graphs
    import random
    random.shuffle(graphs)
    
    model.train()
    for ep in range(epochs):
        ep_loss = 0
        for i in range(0, len(graphs), batch_size):
            batch = graphs[i:i+batch_size]
            b_nodes = torch.tensor(np.array([g['nodes'] for g in batch]), dtype=torch.float32).to(device)
            b_edges = torch.tensor(np.array([g['edges'] for g in batch]), dtype=torch.float32).to(device)
            b_y = torch.tensor([g['target'] for g in batch], dtype=torch.float32).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            y_hat, _ = model(b_nodes, b_edges)
            loss = criterion(y_hat, b_y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(batch)
            
        print(f"Epoch {ep+1}/{epochs} | Loss: {ep_loss/len(graphs):.6f}")
        
    return model

def apply_transgoalnet_inference(df, basic_xt_model, model_checkpoint_path):
    import config
    device = config.DEVICE
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    )
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare data for all dataframe
    df['Trans_xT'] = 0.0
    
    # Needs to match 'prepare_xt_data' basically but inferring directly
    df_mapped = df.copy()
    if 'start_x' not in df_mapped.columns and 'x' in df_mapped.columns:
        df_mapped['start_x'] = df_mapped['x']
        df_mapped['start_y'] = df_mapped['y']
    if 'type' not in df_mapped.columns:
        df_mapped['type'] = 'move'
    if 'result' not in df_mapped.columns:
        df_mapped['result'] = 'success'
    if 'match_id' not in df_mapped.columns:
        df_mapped['match_id'] = 'unknown_match'
    if 'player_name' not in df_mapped.columns and 'player' in df_mapped.columns:
        df_mapped['player_name'] = df_mapped['player']
        
    graphs, _ = prepare_transgoalnet_dataset(df_mapped, basic_xt_model)
    
    with torch.no_grad():
        for batch_i in range(0, len(graphs), 128):
            batch = graphs[batch_i:batch_i+128]
            b_nodes = torch.tensor(np.array([g['nodes'] for g in batch]), dtype=torch.float32).to(device)
            b_edges = torch.tensor(np.array([g['edges'] for g in batch]), dtype=torch.float32).to(device)
            b_y = torch.tensor([g['target'] for g in batch], dtype=torch.float32).unsqueeze(1).to(device)
            
            _, node_embs = model(b_nodes, b_edges)
            
            # calculate specific attribution
            for bi, g in enumerate(batch):
                embs = node_embs[bi] # (MAX_N, H)
                mags = torch.norm(embs, dim=1) # (MAX_N)
                tot_mag = torch.sum(mags) + 1e-6
                
                player_idx = g['player_idx']
                attribution = (mags[player_idx] / tot_mag) * g['target']
                
                # assign to df
                df.loc[g['idx_orig'], 'Trans_xT'] = attribution.item()
                
    return df

def evaluate_transgoalnet(df, basic_xt_model, model_checkpoint_path):
    import config
    import numpy as np
    
    device = config.DEVICE
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    )
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    df_mapped = df.copy()
    if 'start_x' not in df_mapped.columns and 'x' in df_mapped.columns:
        df_mapped['start_x'] = df_mapped['x']
        df_mapped['start_y'] = df_mapped['y']
    if 'type' not in df_mapped.columns:
        df_mapped['type'] = 'move'
    if 'result' not in df_mapped.columns:
        df_mapped['result'] = 'success'
    if 'match_id' not in df_mapped.columns:
        df_mapped['match_id'] = 'unknown_match'
    if 'player_name' not in df_mapped.columns and 'player' in df_mapped.columns:
        df_mapped['player_name'] = df_mapped['player']
        
    graphs, _ = prepare_transgoalnet_dataset(df_mapped, basic_xt_model)
    
    all_targets = []
    all_preds = []
    attentions_list = []
    
    with torch.no_grad():
        for batch_i in range(0, len(graphs), 128):
            batch = graphs[batch_i:batch_i+128]
            b_nodes = torch.tensor(np.array([g['nodes'] for g in batch]), dtype=torch.float32).to(device)
            b_edges = torch.tensor(np.array([g['edges'] for g in batch]), dtype=torch.float32).to(device)
            b_y = torch.tensor([g['target'] for g in batch], dtype=torch.float32).unsqueeze(1).to(device)
            
            y_hat, _ = model(b_nodes, b_edges)
            
            # Get attention from last layer (B, heads, N, N)
            last_layer_attn = getattr(model.layers[-1], 'last_attn', None)
            if last_layer_attn is not None:
                # Average over heads
                avg_attn = last_layer_attn.mean(dim=1) # (B, N, N)
                for bi, g in enumerate(batch):
                    idx = g['player_idx']
                    attn_row = avg_attn[bi, idx, :] # (N,)
                    attentions_list.append(attn_row.max().item())
            
            all_targets.extend(b_y.cpu().numpy().flatten())
            all_preds.extend(y_hat.cpu().numpy().flatten())

    if len(all_targets) == 0:
        return {}

    all_targets = np.nan_to_num(all_targets)
    all_preds = np.nan_to_num(all_preds)

    # 1. Statistical: Pseudo-Brier/MSE
    mse = np.mean((all_targets - all_preds) ** 2)

    # 2. Tactical: Attention Focus
    avg_max_attn = np.mean(attentions_list) if attentions_list else 0.0

    # 3. Stability: Mean Absolute Change
    df_eval = df.copy()
    if 'Trans_xT' not in df_eval.columns:
        df_eval = apply_transgoalnet_inference(df_eval, basic_xt_model, model_checkpoint_path)
    
    mac = 0.0
    if len(df_eval) > 1 and 'Trans_xT' in df_eval.columns:
        df_eval['Trans_xT'] = pd.to_numeric(df_eval['Trans_xT'], errors='coerce')
        valid_mac = df_eval['Trans_xT'].diff().abs().dropna()
        if not valid_mac.empty:
            mac = valid_mac.mean()
        
    # 4. Success: Pearson Correlation
    valid_mask = ~(np.isnan(all_targets) | np.isnan(all_preds))
    if np.sum(valid_mask) > 1 and np.std(all_preds[valid_mask]) > 0 and np.std(all_targets[valid_mask]) > 0:
        pearson_corr = np.corrcoef(all_preds[valid_mask], all_targets[valid_mask])[0, 1]
    else:
        pearson_corr = 0.0

    metrics = {
        "Statistical": {
            "Metric": "Brier Score / MSE",
            "Value": float(mse),
            "Meaning": "Is the probability math 'honest'?"
        },
        "Tactical": {
            "Metric": "Attention Weights (Max focus)",
            "Value": float(avg_max_attn),
            "Meaning": "Is the model looking at the right players/lanes?"
        },
        "Stability": {
            "Metric": "Mean Absolute Change",
            "Value": float(mac),
            "Meaning": "Does the xT value jump around too much?"
        },
        "Success": {
            "Metric": "Pearson Correlation",
            "Value": float(pearson_corr),
            "Meaning": "Does 'high xT' actually lead to more points?"
        }
    }
    return metrics

