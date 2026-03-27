import sys
import os
import torch

framework_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(framework_dir)
sys.path.append(os.path.join(framework_dir, 'src'))

import src.config as config
from src.engine.transgoalnet import TransGoalNet

def debug_load():
    device = config.DEVICE
    model = TransGoalNet(
        node_dim=config.TGN_NODE_DIM, edge_dim=config.TGN_EDGE_DIM, 
        hidden_dim=config.TGN_HIDDEN_DIM, num_heads=config.TGN_NUM_HEADS, 
        num_layers=config.TGN_NUM_LAYERS
    )
    checkpoint_path = os.path.join(config.ASSETS_DIR, 'transgoalnet_global.pth')
    print(f"Loading checkpoint {checkpoint_path} with node_dim={config.TGN_NODE_DIM}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print("Success!")
    except Exception as e:
        print("Error!")
        print(e)

if __name__ == "__main__":
    debug_load()
