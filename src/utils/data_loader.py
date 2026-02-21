import streamlit as st
import pandas as pd
from statsbombpy import sb

@st.cache_data
def load_statsbomb_data(team_name="Hyderabad", limit_matches=5):
    """
    Fetches ISL 2021/22 data from StatsBomb. 
    """
    try:
        # Get Competition
        comps = sb.competitions()
        isl = comps[
            (comps['country_name'] == 'India') | 
            (comps['competition_name'].str.contains('Indian', case=False))
        ]
        
        if isl.empty:
            st.error("ISL 2021/22 data not found.")
            return pd.DataFrame()

        # Get Matches
        matches = sb.matches(competition_id=isl.iloc[0]['competition_id'], 
                             season_id=isl.iloc[0]['season_id'])
        
        # Filter Team
        hfc_matches = matches[(matches['home_team'] == team_name) | 
                              (matches['away_team'] == team_name)]
        
        # Fetch Events (Limit matches for speed)
        all_events = []
        matches_to_process = hfc_matches.head(limit_matches) if limit_matches else hfc_matches
        
        for _, match in matches_to_process.iterrows():
            match_id = match['match_id']
            
            # Determine Result
            if match['home_team'] == team_name:
                res = 'Win' if match['home_score'] > match['away_score'] else \
                      ('Loss' if match['home_score'] < match['away_score'] else 'Draw')
            else:
                res = 'Win' if match['away_score'] > match['home_score'] else \
                      ('Loss' if match['away_score'] < match['home_score'] else 'Draw')
            
            events = sb.events(match_id=match_id)
            events = events[events['team'] == team_name] # Filter for team
            events['match_id'] = match_id
            events['outcome_result'] = res
            all_events.append(events)
            
        if not all_events: return pd.DataFrame()
            
        return pd.concat(all_events, ignore_index=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_passes(df):
    """Filter successful passes and extract coordinates."""
    if df.empty:
        return pd.DataFrame()
        
    # Filter successful passes
    passes = df[(df['type'] == 'Pass') & (df['pass_outcome'].isna())].copy()
    
    # Extract coordinates
    passes[['x', 'y']] = pd.DataFrame(passes['location'].tolist(), index=passes.index)
    passes[['end_x', 'end_y']] = pd.DataFrame(passes['pass_end_location'].tolist(), index=passes.index)
    
    # Rename
    passes.rename(columns={'player': 'player_name', 'pass_recipient': 'pass_recipient_name'}, inplace=True)
    return passes
