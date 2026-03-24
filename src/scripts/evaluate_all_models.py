import sys
import os
import pandas as pd
import numpy as np
import json
import torch
from datetime import datetime
import gc

# Setup paths
framework_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(framework_dir)

from engine.xt_model import ExpectedThreat
from engine.transgoalnet import evaluate_transgoalnet
from engine.metrics import generate_model_evaluation_report
import config
import traceback

def update_progress(status, progress, message):
    try:
        prog_file = os.path.join(config.LOGS_DIR, "evaluation_progress.json")
        with open(prog_file, "w") as f:
            json.dump({"status": status, "progress": progress, "message": message}, f)
    except:
        pass

def main():
    print(f"[{datetime.now()}] Starting Global Model Evaluation Pipeline...")
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
    
    # Check if checkpoints exist
    xt_checkpoint = config.XT_GLOBAL_CHECKPOINT
    trans_checkpoint = config.TGN_GLOBAL_CHECKPOINT
    
    if not os.path.exists(xt_checkpoint) or not os.path.exists(trans_checkpoint):
        msg = "Checkpoints not found. Please run Global Model Training first."
        print(msg)
        update_progress("error", 0.0, msg)
        return

    update_progress("running", 0.15, "Loading previously saved test split...")
    if not os.path.exists(config.TEST_ACTIONS_FILE):
        msg = "Test split dataset not found. Please re-run Global Model Training."
        print(msg)
        update_progress("error", 0.0, msg)
        return
        
    test_actions = pd.read_pickle(config.TEST_ACTIONS_FILE)
    
    if test_actions.empty:
        msg = "Holdout test set is empty. Check dataset."
        print(msg)
        update_progress("error", 0.0, msg)
        return

    test_matches = test_actions['match_id'].unique()

    # Load Models
    update_progress("running", 0.30, "Loading Existing Models...")
    print(f"[{datetime.now()}] Loading Basic xT Model...")
    xt_model = ExpectedThreat.load_checkpoint(xt_checkpoint)
    
    print(f"[{datetime.now()}] Evaluating TransGoalNet on 30% Test Set ({len(test_matches)} matches)...")
    update_progress("running", 0.85, "Running Model Evaluation Loop...")
    
    eval_metrics_dict = evaluate_transgoalnet(test_actions, xt_model, trans_checkpoint)
    
    save_dir_eval = os.path.join(config.LOGS_DIR, "global_tgn_eval.md")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    generate_model_evaluation_report(eval_metrics_dict, save_dir_eval)
    
    del test_actions
    gc.collect()
    
    print(f"[{datetime.now()}] Saved global evaluation report to {save_dir_eval}")
    
    print("Evaluation completed successfully!")
    update_progress("completed", 1.0, "Global TransGoalNet Evaluation finished successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"FATAL ERROR:\n{err_msg}")
        update_progress("error", 0.0, f"Fatal error occurred: {e}")
