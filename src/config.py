import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DNA_DIR = os.path.join(BASE_DIR, 'team_dna')

for d in [DATA_DIR, ASSETS_DIR, LOGS_DIR, DNA_DIR]:
    os.makedirs(d, exist_ok=True)

# File Paths
STYLE_CSS = os.path.join(ASSETS_DIR, 'style.css')
GLOBAL_DATA_FILE = os.path.join(DATA_DIR, 'all_competitions_events.pkl')
TRAIN_ACTIONS_FILE = os.path.join(DATA_DIR, 'train_actions.pkl')
TEST_ACTIONS_FILE = os.path.join(DATA_DIR, 'test_actions.pkl')

# xT Model Parameters
XT_L = 12
XT_W = 8
XT_EPS = 1e-5
XT_CHECKPOINT = os.path.join(ASSETS_DIR, 'xt_checkpoint.npy')
XT_GLOBAL_CHECKPOINT = os.path.join(ASSETS_DIR, 'xt_checkpoint_global.npy')

# TransGoalNet Parameters
TGN_NODE_DIM = 10
TGN_EDGE_DIM = 5
TGN_HIDDEN_DIM = 64
TGN_NUM_HEADS = 4
TGN_NUM_LAYERS = 2
TGN_EPOCHS = 100
TGN_BATCH_SIZE = 64
TGN_LR = 1e-4

TGN_CHECKPOINT = os.path.join(ASSETS_DIR, 'transgoalnet.pth')
TGN_GLOBAL_CHECKPOINT = os.path.join(ASSETS_DIR, 'transgoalnet_global.pth')
TGN_REPORT = os.path.join(ASSETS_DIR, 'transgoalnet_training_report.json')
TGN_ARCH_TXT = os.path.join(ASSETS_DIR, 'transgoalnet_architecture.txt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
