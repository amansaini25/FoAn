import streamlit as st
import pandas as pd
import os
from statsbombpy import sb
from utils.logger import get_logger
import config
import traceback

logger = get_logger()

@st.cache_data
def get_competitions():
    """Fetches all available competitions from StatsBomb."""
    try:
        logger.info("Fetching StatsBomb competitions")
        comps = sb.competitions()
        return comps
    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(f"Error loading competitions:\n{err_msg}")
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
        err_msg = traceback.format_exc()
        logger.error(f"Error loading matches:\n{err_msg}")
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
            
        full_df = pd.concat(all_events, ignore_index=True)
        
        # Save raw data chunk locally
        sanitized_team = team_name.replace(' ', '_')
        save_file = os.path.join(config.DATA_DIR, f"{sanitized_team}_raw_events.csv")
        full_df.to_csv(save_file, index=False)
        logger.info(f"Saved {len(full_df)} events for {team_name} to {save_file}")
            
        logger.info("Successfully fetched and compiled match events.")
        return full_df

    except Exception as e:
        err_msg = traceback.format_exc()
        logger.error(f"Error loading event data from StatsBomb:\n{err_msg}")
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

def load_all_training_data(progress_bar, status_text):
    """
    Fetches events from ALL StatsBomb matches and caches them locally, 
    updating UI elements with progress.
    """
    safe_file = config.GLOBAL_DATA_FILE
    if os.path.exists(safe_file):
        status_text.text("Loading saved global data (found locally)...")
        progress_bar.progress(1.0)
        return pd.read_pickle(safe_file)
        
    status_text.text("Fetching all competitions...")
    try:
        comps = sb.competitions()
    except Exception as e:
        st.error(f"Failed fetching competitions: {e}")
        return pd.DataFrame()
        
    all_events = []
    total_comps = len(comps)
    
    for i, (_, comp) in enumerate(comps.iterrows()):
        comp_id = comp['competition_id']
        season_id = comp['season_id']
        status_text.text(f"Fetching matches for Comp: {comp_id}, Season: {season_id} ({i+1}/{total_comps})")
        
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches.empty: continue
            
            # For each match, load events and filter for only what's needed for xT models
            for _, match in matches.iterrows():
                try:
                    events = sb.events(match_id=match['match_id'])
                    # Keep minimal essential columns/rows to save RAM
                    events = events[events['type'].isin(['Pass', 'Shot', 'Carry'])].copy()
                    events['match_id'] = match['match_id']
                    if not events.empty:
                        all_events.append(events)
                except:
                    continue
        except Exception as e:
            continue
            
        progress_bar.progress((i + 1) / total_comps)
        
    if all_events:
        status_text.text("Saving compiled dataset to local storage...")
        full_df = pd.concat(all_events, ignore_index=True)
        full_df.to_pickle(safe_file)
        status_text.text("Global dataset compiled successfully!")
        return full_df
        
    status_text.text("No events found!")
    return pd.DataFrame()

