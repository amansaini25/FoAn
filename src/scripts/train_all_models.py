import sys
import os
import pandas as pd
import numpy as np
import json
import torch
from statsbombpy import sb
from datetime import datetime
import gc


# Setup paths
framework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(framework_dir)

from engine.xt_model import ExpectedThreat, prepare_xt_data
from engine.transgoalnet import train_transgoalnet, prepare_transgoalnet_dataset, TransGoalNet
from utils.data_loader import load_all_training_data
import config
import traceback

def update_progress(status, progress, message):
    try:
        prog_file = os.path.join(config.LOGS_DIR, "training_progress.json")
        with open(prog_file, "w") as f:
            json.dump({"status": status, "progress": progress, "message": message}, f)
    except:
        pass

def main():
    print(f"[{datetime.now()}] Starting Full Dataset Training Pipeline...")
    
    update_progress("running", 0.05, "Initializing Data Fetching Pipeline...")
    
    class MockerUI:
        def __init__(self):
            self.msg = ""
            self.val = 0.05
        def progress(self, val):
            self.val = 0.05 + (0.5 * val)
            update_progress("running", self.val, self.msg)
        def text(self, msg):
            self.msg = msg
            print(f"[{datetime.now()}] {msg}")
            update_progress("running", self.val, self.msg)

    ui_mocker = MockerUI()
    training_raw_df = load_all_training_data(ui_mocker, ui_mocker)
    
    if training_raw_df.empty:
        msg = "No events gathered! Exiting."
        print(msg)
        update_progress("error", 0.0, msg)
        return
        
    dataset_info = {
        "total_events_loaded": len(training_raw_df),
        "total_matches_processed": training_raw_df['match_id'].nunique() if 'match_id' in training_raw_df.columns else 0
    }
    
    print(f"Dataset compiled. Total useful events: {len(training_raw_df)}")
    
    assets_dir = os.path.join(framework_dir, '..', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # 2. Train Basic xT
    update_progress("running", 0.60, "Fitting Basic ExpectedThreat Model...")
    print(f"[{datetime.now()}] Preparing Basic xT Data...")
    actions_df = prepare_xt_data(training_raw_df)
    
    dataset_info["total_actions_for_xt"] = len(actions_df)
    
    del training_raw_df
    gc.collect()

    print(f"[{datetime.now()}] Fitting Basic ExpectedThreat Model...")
    xt_model = ExpectedThreat(l=config.XT_L, w=config.XT_W, eps=config.XT_EPS)
    xt_model.fit(actions_df)
    xt_checkpoint = config.XT_GLOBAL_CHECKPOINT
    xt_model.save_checkpoint(xt_checkpoint)
    print(f"[{datetime.now()}] Saved Basic xT Model.")
    
    # Split actions_df by match_id (70% Train / 30% Test)
    unique_matches = actions_df['match_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_matches)
    split_idx = int(len(unique_matches) * 0.7)
    train_matches = unique_matches[:split_idx]
    test_matches = unique_matches[split_idx:]
    
    train_actions = actions_df[actions_df['match_id'].isin(train_matches)].copy()
    test_actions = actions_df[actions_df['match_id'].isin(test_matches)].copy()
    
    dataset_info["train_matches_count"] = len(train_matches)
    dataset_info["test_matches_count"] = len(test_matches)
    
    print(f"[{datetime.now()}] Saving Train and Test Splits locally for evaluation...")
    train_actions.to_pickle(config.TRAIN_ACTIONS_FILE)
    test_actions.to_pickle(config.TEST_ACTIONS_FILE)
    
    # 3. Train TransGoalNet
    update_progress("running", 0.7, "Preparing TransGoalNet Graphics Dataset...")
    print(f"[{datetime.now()}] Preparing TransGoalNet Training Dataset...")
    graphs, max_n = prepare_transgoalnet_dataset(train_actions, xt_model)
    dataset_info["train_event_graphs_built"] = len(graphs)
    
    # Free Heavy Variables from memory
    del actions_df
    del train_actions
    gc.collect()
    
    device = config.DEVICE
    msg2 = f"Training TransGoalNet on {device} ({len(graphs)} graphs)..."
    update_progress("running", 0.75, msg2)
    print(f"[{datetime.now()}] {msg2}")
    
    # Modified training to return metrics
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TGN_LR)
    criterion = torch.nn.MSELoss()
    
    epochs = config.TGN_EPOCHS
    batch_size = config.TGN_BATCH_SIZE
    losses = []
    
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
            
        avg_loss = ep_loss/len(graphs)
        losses.append(avg_loss)
        
        ep_msg = f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.6f}"
        print(ep_msg)
        update_progress("running", 0.75 + 0.25 * ((ep+1)/epochs), f"Training TransGoalNet: {ep_msg}")
        
    trans_checkpoint = config.TGN_GLOBAL_CHECKPOINT
    torch.save(model.state_dict(), trans_checkpoint)
    print(f"[{datetime.now()}] Saved TransGoalNet Model.")
    
    # 3.5 Evaluate on 30% Test Data
    if len(test_actions) > 0:
        print(f"[{datetime.now()}] Evaluating TransGoalNet on 30% Test Set ({len(test_matches)} matches)...")
        update_progress("running", 0.9, "Evaluating TransGoalNet on Test Data...")
        from engine.transgoalnet import evaluate_transgoalnet
        from engine.metrics import generate_model_evaluation_report
        
        eval_metrics_dict = evaluate_transgoalnet(test_actions, xt_model, trans_checkpoint)
        eval_metrics["test_set_performance"] = eval_metrics_dict
        
        save_dir_eval = os.path.join(config.LOGS_DIR, "global_tgn_eval.md")
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        generate_model_evaluation_report(eval_metrics_dict, save_dir_eval)
        print(f"[{datetime.now()}] Saved global evaluation report to {save_dir_eval}")
    
    # 4. Save Info & Architecture
    eval_metrics = {
        "training_epochs": epochs,
        "final_mse_loss": losses[-1],
        "loss_history": losses
    }
    
    architecture = str(model)
    
    report = {
        "timestamp": str(datetime.now()),
        "dataset_info": dataset_info,
        "evaluation_metrics": eval_metrics,
        "architecture": architecture
    }
    
    report_path = config.TGN_REPORT
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    # Also save architecture to a plain text file for easy reading
    arch_path = config.TGN_ARCH_TXT
    with open(arch_path, 'w') as f:
        f.write("TransGoalNet PyTorch Architecture\n")
        f.write("=================================\n\n")
        f.write(architecture)
        
    print(f"[{datetime.now()}] Saved Training Report to {report_path} and {arch_path}.")
    print("All tasks completed successfully!")
    update_progress("completed", 1.0, "All TransGoalNet Models successfully trained and exported!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"FATAL ERROR:\n{err_msg}")
        update_progress("error", 0.0, f"Fatal error occurred: {e}")
