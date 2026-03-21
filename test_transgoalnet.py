import sys
import os

sys.path.append(r"d:\MSC DS ABDN\PROJ_The science of football - Jan 2026\CODE\Vibe\football_analytics_framework\src")

try:
    from engine.transgoalnet import TransGoalNet
    import torch
    model = TransGoalNet()
    print("TransGoalNet initialized successfully. Device:", 'cuda' if torch.cuda.is_available() else 'cpu')
except Exception as e:
    import traceback
    traceback.print_exc()
