import numpy as np
import pandas as pd

def get_xt_grid():
    """Returns a 12x8 simplified xT grid."""
    return np.array([
        [0.006, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.006],
        [0.007, 0.009, 0.010, 0.010, 0.010, 0.010, 0.009, 0.007],
        [0.008, 0.010, 0.011, 0.012, 0.012, 0.011, 0.010, 0.008],
        [0.009, 0.011, 0.013, 0.014, 0.014, 0.013, 0.011, 0.009],
        [0.010, 0.013, 0.015, 0.017, 0.017, 0.015, 0.013, 0.010],
        [0.012, 0.015, 0.018, 0.021, 0.021, 0.018, 0.015, 0.012],
        [0.014, 0.018, 0.022, 0.026, 0.026, 0.022, 0.018, 0.014],
        [0.017, 0.022, 0.028, 0.035, 0.035, 0.028, 0.022, 0.017],
        [0.021, 0.028, 0.038, 0.050, 0.050, 0.038, 0.028, 0.021],
        [0.027, 0.038, 0.055, 0.080, 0.080, 0.055, 0.038, 0.027],
        [0.035, 0.055, 0.085, 0.140, 0.140, 0.085, 0.055, 0.035],
        [0.045, 0.075, 0.120, 0.250, 0.250, 0.120, 0.075, 0.045]
    ]).T

def calculate_xt(row, grid, x_bins=12, y_bins=8, pitch_length=120, pitch_width=80):
    """Calculates the xT for a single pass row."""
    try:
        start_x = min(int(row['x'] / (pitch_length/x_bins)), x_bins-1)
        start_y = min(int(row['y'] / (pitch_width/y_bins)), y_bins-1)
        end_x = min(int(row['end_x'] / (pitch_length/x_bins)), x_bins-1)
        end_y = min(int(row['end_y'] / (pitch_width/y_bins)), y_bins-1)
        return grid[end_y, end_x] - grid[start_y, start_x]
    except (KeyError, TypeError):
        return 0.0

def apply_xt_to_passes(pass_df):
    """Applies xT calculation to a dataframe of passes."""
    if pass_df.empty:
        return pass_df
    xt_grid = get_xt_grid()
    pass_df['xT'] = pass_df.apply(lambda x: calculate_xt(x, xt_grid), axis=1)
    return pass_df
