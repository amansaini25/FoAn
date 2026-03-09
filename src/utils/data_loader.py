import streamlit as st
import pandas as pd
from statsbombpy import sb
from utils.logger import get_logger

logger = get_logger()

@st.cache_data
def get_competitions():
    """Fetches all available competitions from StatsBomb."""
    try:
        logger.info("Fetching StatsBomb competitions")
        comps = sb.competitions()
        return comps
    except Exception as e:
        logger.exception(f"Error loading competitions: {e}")
        st.error(f"Error loading competitions: {e}")
        return pd.DataFrame()

@st.cache_data
def get_matches(competition_id, season_id):
    """Fetches matches for a specific competition and season."""
    try:
        logger.info(f"Fetching matches for comp: {competition_id}, season: {season_id}")
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        return matches
    except Exception as e:
        logger.exception(f"Error loading matches: {e}")
        st.error(f"Error loading matches: {e}")
        return pd.DataFrame()

@st.cache_data
def load_statsbomb_data(matches_df, team_name, limit_matches=None, filter_team=True):
    """
    Fetches events from StatsBomb for the provided matches.
    """
    try:
        logger.info(f"Loading StatsBomb events for Team: {team_name}, matches: {len(matches_df)}")
        
        if matches_df.empty:
            logger.warning("Empty matches dataframe provided.")
            return pd.DataFrame()

        # Fetch Events (Limit matches for speed)
        all_events = []
        matches_to_process = matches_df.head(limit_matches) if limit_matches else matches_df

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
            if filter_team:
                events = events[events['team'] == team_name] # Filter for team
            events['match_id'] = match_id
            events['outcome_result'] = res
            all_events.append(events)
            
        if not all_events: 
            logger.warning("No events found for the specified matches.")
            return pd.DataFrame()
            
        logger.info("Successfully fetched and compiled match events.")
        return pd.concat(all_events, ignore_index=True)

    except Exception as e:
        logger.exception(f"Error loading event data from StatsBomb: {e}")
        st.error(f"Error loading event data: {e}")
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
