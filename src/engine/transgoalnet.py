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
    def __init__(self, node_dim=10, edge_dim=5, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
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

def prepare_transgoalnet_dataset(actions_df, basic_xt_model, k_window=20):
    """ Builds Graph nodes and edges for batches using sliding temporal window. """
    actions_df = actions_df.copy()
    if 'xT' not in actions_df.columns:
        actions_df['xT'] = basic_xt_model.rate(actions_df).fillna(0.0)
    
    # Needs chronological sort for sliding window
    if 'timestamp' in actions_df.columns:
        actions_df = actions_df.sort_values(['match_id', 'timestamp'])
    
    graphs = []
    
    for match_id, match_df in actions_df.groupby('match_id'):
        players = match_df['player_name'].unique().tolist()
        player_to_idx = {p: i for i, p in enumerate(players) if not pd.isna(p)}
        N = len(player_to_idx)
        
        if N == 0: continue
        
        # Simple continuous node feature compilation based on previous k events
        # We will iterate through events and maintain a rolling window
        
        MAX_N = 30
        
        # We need a way to store historical actions.
        match_events = match_df.to_dict('records')
        num_events = len(match_events)
        
        # Precompute player actions for faster lookup
        player_actions_idx = {p: [] for p in player_to_idx.keys()}
        for i, row in enumerate(match_events):
            p = row.get('player_name')
            if p in player_actions_idx:
                player_actions_idx[p].append(i)
                
        if match_id % 10 == 0:
            print(f"  -> Processing match {match_id} | {num_events} events")
            
        for i, row in enumerate(match_events):
            if pd.isna(row['player_name']) or row['player_name'] not in player_to_idx:
                continue
            idx = player_to_idx[row['player_name']]
            if idx >= MAX_N: continue
            
            # Temporal Window: Get previous k events
            start_idx = max(0, i - k_window)
            window_events = match_events[start_idx:i+1] # Include current for stats
            
            # 1. Calculate 10 Node Features for ALL players IN THIS MATCH history
            node_feats = np.zeros((MAX_N, 10))
            
            # History up to i
            for p, p_idx in player_to_idx.items():
                if p_idx >= MAX_N: continue
                
                # Get indices of this player's actions that happened before or AT `i`
                p_history_indices = [idx for idx in player_actions_idx[p] if idx <= i]
                if not p_history_indices: continue
                
                # We can quickly aggregate based on row fields instead of DataFrame filters
                goals, dribbles, tackles, total_passes, acc_passes, key_passes = 0, 0, 0, 0, 0, 0
                interceptions, clearances = 0, 0
                total_xt = 0.0
                shots = 0
                
                for h_idx in p_history_indices:
                    h_row = match_events[h_idx]
                    t = str(h_row.get('type', '')).lower()
                    r = str(h_row.get('result', '')).lower()
                    val_xt = h_row.get('xT', 0.0)
                    if pd.isna(val_xt): val_xt = 0.0
                    
                    if t == 'shot':
                        shots += 1
                        if r == 'goal': goals += 1
                    elif t == 'dribble' and r == 'success':
                        dribbles += 1
                    elif 'tackle' in t:
                        tackles += 1
                    elif 'interception' in t:
                        interceptions += 1
                    elif 'clearance' in t:
                        clearances += 1
                    elif 'pass' in t:
                        total_passes += 1
                        if r == 'success': acc_passes += 1
                        if val_xt > 0.05: key_passes += 1
                        
                    total_xt += val_xt
                
                acc_pass_pct = acc_passes / total_passes if total_passes > 0 else 0.0
                rating = total_xt / len(p_history_indices) if len(p_history_indices) > 0 else 0.0
                goal_conv = goals / shots if shots > 0 else 0.0
                
                node_feats[p_idx, 0] = goals
                node_feats[p_idx, 1] = dribbles
                node_feats[p_idx, 2] = tackles
                node_feats[p_idx, 3] = acc_pass_pct
                node_feats[p_idx, 4] = rating
                node_feats[p_idx, 5] = goal_conv
                node_feats[p_idx, 6] = interceptions
                node_feats[p_idx, 7] = clearances
                node_feats[p_idx, 8] = acc_passes
                node_feats[p_idx, 9] = key_passes
                
            # Positional Encodings (Add average spatial location to node features? Paper says added to attributes)
            # The paper says d=10, so node_dim=10. Positional encodings are usually added directly to the embeddings later, 
            # but we can append spatial info or transform them. Let's keep node_dim=10 purely statistical as per paper,
            # and we will add positional encoding inside the network forward pass.
            
            # Edge features bias (MAX_N x MAX_N x 5)
            # For the window, edges represent passes/interactions within the temporal window
            edge_feats = np.zeros((MAX_N, MAX_N, 5))
            
            for w_row in window_events:
                p_name = w_row.get('player_name')
                if pd.isna(p_name) or p_name not in player_to_idx:
                    continue
                w_p_idx = player_to_idx[p_name]
                if w_p_idx >= MAX_N: continue
                
                # Self connection for actions
                edge_feats[w_p_idx, w_p_idx, 0] = float(w_row.get('start_x', 0) or 0) / 120.0
                edge_feats[w_p_idx, w_p_idx, 1] = float(w_row.get('start_y', 0) or 0) / 80.0
                edge_feats[w_p_idx, w_p_idx, 2] = float(w_row.get('end_x', 0) or 0) / 120.0
                edge_feats[w_p_idx, w_p_idx, 3] = float(w_row.get('end_y', 0) or 0) / 80.0
                
                val_xt = w_row.get('xT', 0)
                edge_feats[w_p_idx, w_p_idx, 4] = float(val_xt) if not pd.isna(val_xt) else 0.0
                
                # If there's a recipient (pass), create directed edge
                rec_name = w_row.get('recipient_name')
                if rec_name and not pd.isna(rec_name) and rec_name in player_to_idx:
                    rec_idx = player_to_idx[rec_name]
                    if rec_idx < MAX_N:
                        edge_feats[w_p_idx, rec_idx] = edge_feats[w_p_idx, w_p_idx]
            
            curr_xt = float(row.get('xT', 0.0) if not pd.isna(row.get('xT', 0.0)) else 0.0)
            delta_xt = curr_xt
            if i > 0:
                prev_row = match_events[i-1]
                prev_xt = float(prev_row.get('xT', 0.0) if not pd.isna(prev_row.get('xT', 0.0)) else 0.0)
                
                curr_team = row.get('team')
                prev_team = prev_row.get('team')
                
                if curr_team == prev_team:
                    delta_xt = curr_xt - prev_xt
                else:
                    # Possession changed: threat swing
                    delta_xt = curr_xt + prev_xt
                    
            graphs.append({
                'nodes': node_feats,
                'edges': edge_feats,
                'target': delta_xt,
                'player_idx': idx,
                'match_id': row['match_id'],
                'idx_orig': row.get('idx', i),
                'idx_to_player': {idx_val: p_name for p_name, idx_val in player_to_idx.items()}
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
    
    # Paper specifics: Weight Decay, StepLR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Shuffle graphs
    import random
    random.shuffle(graphs)
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    train_history = {"epochs": [], "loss": [], "lr": []}
    log_path = os.path.join(config.LOGS_DIR, "tgn_training_log.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
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
            
        avg_loss = ep_loss / len(graphs)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
        scheduler.step()
        
        train_history['epochs'].append(ep + 1)
        train_history['loss'].append(avg_loss)
        train_history['lr'].append(current_lr)
        
        with open(log_path, 'w') as f:
            import json
            json.dump(train_history, f)
        
        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered at epoch {ep+1} due to no improvement in training loss.")
            break
            
        # Periodic Checkpointing
        if ep + 1 > 15 and (ep + 1) % 5 == 0:
            import os
            from datetime import datetime
            ckpt_dir = os.path.join(config.ASSETS_DIR, 'tgn_checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = f"tgn_ep{ep+1}_loss{avg_loss:.4f}_{timestamp}.pth"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_name}")
        
    return model

def apply_transgoalnet_inference(df, basic_xt_model, model_checkpoint_path):
    import config
    device = config.DEVICE
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    )
    # Load to CPU first to prevent CUDA out-of-memory or threading crashes in Streamlit
    model.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu'))
    model.to(device)
    model.eval()
    
    # Prepare data for all dataframe
    df['Trans_xT'] = 0.0
    
    # Needs to match 'prepare_xt_data' basically but inferring directly
    df_mapped = df.copy()
    # Force type and result so xT translates properly
    df_mapped['type'] = 'move'
    df_mapped['result'] = 'success'
    
    if 'start_x' not in df_mapped.columns and 'x' in df_mapped.columns:
        df_mapped['start_x'] = df_mapped['x']
        df_mapped['start_y'] = df_mapped['y']
    if 'match_id' not in df_mapped.columns:
        df_mapped['match_id'] = 'unknown_match'
    if 'player_name' not in df_mapped.columns and 'player' in df_mapped.columns:
        df_mapped['player_name'] = df_mapped['player']
        
    graphs, max_n_val = prepare_transgoalnet_dataset(df_mapped, basic_xt_model)
    
    max_attn_val = -1.0
    top_lane = {'passer': None, 'recipient': None, 'attention': 0.0}
    
    with torch.no_grad():
        for batch_i in range(0, len(graphs), 128):
            batch = graphs[batch_i:batch_i+128]
            b_nodes = torch.tensor(np.array([g['nodes'] for g in batch]), dtype=torch.float32).to(device)
            b_edges = torch.tensor(np.array([g['edges'] for g in batch]), dtype=torch.float32).to(device)
            b_y = torch.tensor([g['target'] for g in batch], dtype=torch.float32).unsqueeze(1).to(device)
            
            y_hat, node_embs = model(b_nodes, b_edges)
            
            last_layer_attn = getattr(model.layers[-1], 'last_attn', None)
            if last_layer_attn is not None:
                max_attn_batch = last_layer_attn.max(dim=1)[0].cpu().numpy() # [B, N, N]
                for bi in range(len(batch)):
                    # find absolute max in this subgraph
                    # shape is (N, N)
                    N_sub = max_attn_batch[bi].shape[0]
                    # avoid self-attention diagonal
                    np.fill_diagonal(max_attn_batch[bi], 0)
                    if max_attn_batch[bi].max() > max_attn_val:
                        max_local = max_attn_batch[bi].max()
                        idx = np.unravel_index(np.argmax(max_attn_batch[bi]), max_attn_batch[bi].shape)
                        max_attn_val = max_local
                        
                        p1_idx, p2_idx = idx
                        local_idx_to_player = batch[bi].get('idx_to_player', {})
                        top_lane = {
                            'passer': local_idx_to_player.get(p1_idx, "Unknown"),
                            'recipient': local_idx_to_player.get(p2_idx, "Unknown"),
                            'attention': float(max_attn_val)
                        }
            
            # extract predictions
            for bi, g in enumerate(batch):
                pred_xt = y_hat[bi].item()
                
                # assign the predicted change in xT directly to the dataframe
                df.loc[g['idx_orig'], 'Trans_xT'] = pred_xt
                
    return df, top_lane

def evaluate_transgoalnet(df, basic_xt_model, model_checkpoint_path):
    import config
    import numpy as np
    
    device = config.DEVICE
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    )
    # Load to CPU first to prevent CUDA out-of-memory or threading crashes in Streamlit
    model.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu'))
    model.to(device)
    model.eval()
    
    df_mapped = df.copy()
    # Force type and result so xT translates properly
    df_mapped['type'] = 'move'
    df_mapped['result'] = 'success'
    
    if 'start_x' not in df_mapped.columns and 'x' in df_mapped.columns:
        df_mapped['start_x'] = df_mapped['x']
        df_mapped['start_y'] = df_mapped['y']
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

