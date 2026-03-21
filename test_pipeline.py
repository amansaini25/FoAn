import sys
import os
import pandas as pd

sys.path.append(r"d:\MSC DS ABDN\PROJ_The science of football - Jan 2026\CODE\Vibe\football_analytics_framework\src")

try:
    from engine.transgoalnet import train_transgoalnet, prepare_transgoalnet_dataset, apply_transgoalnet_inference
    from engine.xt_model import ExpectedThreat
    from engine.metrics import calculate_team_dna

    # Dummy dataframe based on what statsbomb outputs
    df = pd.DataFrame([{
        'match_id': 1,
        'minute': 12,
        'second': 30,
        'player_name': 'Player A',
        'type': 'move',
        'result': 'success',
        'start_x': 50, 'start_y': 50,
        'end_x': 60, 'end_y': 60
    }])
    
    # Needs a basic_xt_model to work
    xt = ExpectedThreat()
    xt.xT = __import__('numpy').zeros((8, 12))
    
    print("Preparing dataset...")
    graphs, max_n = prepare_transgoalnet_dataset(df, xt)
    print("Graphs created:", len(graphs))
    
    print("Training dummy model...")
    model = train_transgoalnet(graphs, max_n, epochs=1, device='cpu')
    print("Done training.")
    
    # Save dummy model
    import torch
    torch.save(model.state_dict(), 'dummy_trans.pth')
    
    print("Inference dummy model...")
    df_out = apply_transgoalnet_inference(df, xt, 'dummy_trans.pth')
    print("Inference output columns:", list(df_out.columns))
    
    print("Team DNA metrics:")
    metrics = calculate_team_dna(df_out)
    print(metrics)
    
except Exception as e:
    import traceback
    traceback.print_exc()
