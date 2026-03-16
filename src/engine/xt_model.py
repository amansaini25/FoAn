import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger()

def _get_cell_indexes(x, y, l=12, w=8):
    xi = (x / 120.0) * l
    yj = (y / 80.0) * w
    xi = xi.astype(int).clip(0, l - 1)
    yj = yj.astype(int).clip(0, w - 1)
    return xi, yj

def _get_flat_indexes(x, y, l=12, w=8):
    xi, yj = _get_cell_indexes(x, y, l, w)
    return yj * l + xi

def _count(x, y, l=12, w=8):
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    if x.empty or y.empty:
        return np.zeros((w, l), dtype=int)
    flat_indexes = _get_flat_indexes(x, y, l, w)
    vc = flat_indexes.value_counts(sort=False)
    vector = np.zeros(w * l, dtype=int)
    vector[vc.index] = vc
    return vector.reshape((w, l))

def _safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

def prepare_xt_data(raw_df):
    """Converts StatsBomb raw dataframe into a standard actions dataframe for xT."""
    actions = []
    
    for _, row in raw_df.iterrows():
        action_type = row.get('type')
        if pd.isna(action_type):
            continue
            
        loc = row.get('location')
        if not isinstance(loc, list) or len(loc) < 2:
            continue
            
        start_x, start_y = loc[0], loc[1]
        
        if action_type == 'Shot':
            result = 'success' if row.get('shot_outcome') == 'Goal' else 'fail'
            actions.append({
                'type': 'shot',
                'result': result,
                'start_x': start_x, 'start_y': start_y,
                'end_x': start_x, 'end_y': start_y
            })
        elif action_type == 'Pass':
            end_loc = row.get('pass_end_location')
            if isinstance(end_loc, list) and len(end_loc) >= 2:
                result = 'fail' if pd.notna(row.get('pass_outcome')) else 'success'
                actions.append({
                    'type': 'move',
                    'result': result,
                    'start_x': start_x, 'start_y': start_y,
                    'end_x': end_loc[0], 'end_y': end_loc[1]
                })
        elif action_type == 'Carry':
            end_loc = row.get('carry_end_location')
            if isinstance(end_loc, list) and len(end_loc) >= 2:
                # Carries are generally successful moves
                actions.append({
                    'type': 'move',
                    'result': 'success',
                    'start_x': start_x, 'start_y': start_y,
                    'end_x': end_loc[0], 'end_y': end_loc[1]
                })
                
    return pd.DataFrame(actions)

class ExpectedThreat:
    """An implementation of the Expected Threat (xT) model using dynamic programming."""
    def __init__(self, l=12, w=8, eps=1e-5):
        self.l = l
        self.w = w
        self.eps = eps
        self.heatmaps = []
        self.xT = np.zeros((self.w, self.l))
        self.scoring_prob_matrix = None
        self.shot_prob_matrix = None
        self.move_prob_matrix = None
        self.transition_matrix = None

    @classmethod
    def load_checkpoint(cls, filepath, l=12, w=8, eps=1e-5):
        import os
        if not os.path.exists(filepath):
            return None
        model = cls(l=l, w=w, eps=eps)
        model.xT = np.load(filepath)
        return model

    def save_checkpoint(self, filepath):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.xT)

    def fit(self, actions: pd.DataFrame):
        shots = actions[actions['type'] == 'shot']
        goals = shots[shots['result'] == 'success']
        moves = actions[actions['type'] == 'move']
        
        shotmatrix = _count(shots['start_x'], shots['start_y'], self.l, self.w)
        goalmatrix = _count(goals['start_x'], goals['start_y'], self.l, self.w)
        self.scoring_prob_matrix = _safe_divide(goalmatrix, shotmatrix)

        movematrix = _count(moves['start_x'], moves['start_y'], self.l, self.w)
        totalmatrix = movematrix + shotmatrix
        self.shot_prob_matrix = _safe_divide(shotmatrix, totalmatrix)
        self.move_prob_matrix = _safe_divide(movematrix, totalmatrix)

        X = pd.DataFrame()
        X["start_cell"] = _get_flat_indexes(moves['start_x'], moves['start_y'], self.l, self.w)
        X["end_cell"] = _get_flat_indexes(moves['end_x'], moves['end_y'], self.l, self.w)
        X["result"] = moves['result']

        vc = X["start_cell"].value_counts(sort=False)
        start_counts = np.zeros(self.w * self.l)
        start_counts[vc.index] = vc

        self.transition_matrix = np.zeros((self.w * self.l, self.w * self.l))
        for i in range(0, self.w * self.l):
            vc2 = X[(X["start_cell"] == i) & (X["result"] == 'success')]["end_cell"].value_counts(sort=False)
            if start_counts[i] > 0:
                self.transition_matrix[i, vc2.index] = vc2 / start_counts[i]

        self.xT = np.zeros((self.w, self.l))
        self._solve()
        return self

    def _solve(self):
        gs = self.scoring_prob_matrix * self.shot_prob_matrix
        diff = np.ones((self.w, self.l), dtype=np.float64)
        it = 0
        self.heatmaps.append(self.xT.copy())

        while np.any(diff > self.eps):
            # Fast vectorized payoff calculation
            total_payoff = np.dot(self.transition_matrix, self.xT.flatten()).reshape((self.w, self.l))
            
            newxT = gs + (self.move_prob_matrix * total_payoff)
            diff = newxT - self.xT
            self.xT = newxT
            self.heatmaps.append(self.xT.copy())
            it += 1
            if it > 5:
                logger.warning("xT solver stopped after 1000 iterations.")
                break

    def rate(self, actions: pd.DataFrame):
        ratings = pd.Series(np.nan, index=actions.index)

        # Filter out actions that aren't successful moves
        valid_idx = (actions['type'] == 'move') & (actions['result'] == 'success')
        move_actions = actions[valid_idx].copy()
        
        if move_actions.empty:
            return ratings

        startxc, startyc = _get_cell_indexes(move_actions['start_x'], move_actions['start_y'], self.l, self.w)
        endxc, endyc = _get_cell_indexes(move_actions['end_x'], move_actions['end_y'], self.l, self.w)

        xT_start = self.xT[startyc, startxc]
        xT_end = self.xT[endyc, endxc]

        ratings.loc[move_actions.index] = xT_end - xT_start
        return ratings

def apply_xt_to_passes(pass_df, xt_model):
    """Applies xT calculation to a dataframe of passes using an ExpectedThreat model."""
    if pass_df.empty:
        return pass_df

    # Adapt pass_df for the model's rate function
    actions_for_rating = pd.DataFrame({
        'type': 'move',
        'result': 'success',
        'start_x': pass_df['x'],
        'start_y': pass_df['y'],
        'end_x': pass_df['end_x'],
        'end_y': pass_df['end_y']
    }, index=pass_df.index)
    
    pass_df['xT'] = xt_model.rate(actions_for_rating)
    # Fill NaN with 0 for passes that might not have a rating (e.g. failing validation)
    pass_df['xT'] = pass_df['xT'].fillna(0.0)
    return pass_df
