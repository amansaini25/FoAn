import streamlit as st
import pandas as pd
import os
from statsbombpy import sb
from utils.logger import get_logger
import config
import traceback

logger = get_logger()

TEAM_MAPPING = {
    "Chennaiyin": "Chennaiyin FC"
    # Can be extended for other teams if needed
}

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
        
        if not matches.empty:
            if 'home_team' in matches.columns:
                matches['home_team'] = matches['home_team'].replace(TEAM_MAPPING)
            if 'away_team' in matches.columns:
                matches['away_team'] = matches['away_team'].replace(TEAM_MAPPING)
                
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

        # Offline File Check
        sanitized_team = team_name.replace(' ', '_')
        save_file = os.path.join(config.DATA_DIR, f"{sanitized_team}_raw_events.csv")
        
        df_cache = pd.DataFrame()
        if os.path.exists(save_file):
            logger.info(f"Loaded Offline Component successfully for {team_name} from {save_file}")
            df_cache = pd.read_csv(save_file)
            
            if not df_cache.empty and 'team' in df_cache.columns:
                df_cache['team'] = df_cache['team'].replace(TEAM_MAPPING)
                
        # Find missing matches
        if limit_matches:
            match_ids_to_keep = matches_df.head(limit_matches)['match_id'].tolist()
        else:
            match_ids_to_keep = matches_df['match_id'].tolist()
            
        missing_matches = match_ids_to_keep
        if not df_cache.empty and 'match_id' in df_cache.columns:
            cached_match_ids = df_cache['match_id'].unique()
            missing_matches = [m_id for m_id in match_ids_to_keep if m_id not in cached_match_ids]

        # Fetch Events for missing matches
        if missing_matches:
            logger.info(f"Fetching {len(missing_matches)} missing matches for {team_name} from StatsBomb API...")
            all_events = []
            matches_to_process = matches_df[matches_df['match_id'].isin(missing_matches)]

            for _, match in matches_to_process.iterrows():
                match_id = match['match_id']

                # Determine Result
                if match['home_team'] == team_name:
                    res = 'Win' if match['home_score'] > match['away_score'] else \
                          ('Loss' if match['home_score'] < match['away_score'] else 'Draw')
                else:
                    res = 'Win' if match['away_score'] > match['home_score'] else \
                          ('Loss' if match['away_score'] < match['home_score'] else 'Draw')

                try:
                    events = sb.events(match_id=match_id)
                except Exception as e:
                    logger.warning(f"Could not fetch events for match {match_id}: {e}")
                    continue
                    
                if not events.empty and 'team' in events.columns:
                    events['team'] = events['team'].replace(TEAM_MAPPING)
                    
                # We save all events for the match in the cache, not just the team's, so don't filter here
                events['match_id'] = match_id
                events['outcome_result'] = res
                all_events.append(events)
                
            if all_events:
                new_events_df = pd.concat(all_events, ignore_index=True)
                if df_cache.empty:
                    df_cache = new_events_df
                else:
                    df_cache = pd.concat([df_cache, new_events_df], ignore_index=True)
                    
                df_cache.to_csv(save_file, index=False)
                logger.info(f"Saved {len(df_cache)} events for {team_name} to {save_file}")
            elif not missing_matches and not df_cache.empty:
                logger.info("All matches already in cache.")
            else:
                logger.warning("No new events found for the specified matches.")
                
        if not df_cache.empty:
            filtered_cache = df_cache[df_cache['match_id'].isin(match_ids_to_keep)].copy()
            if filter_team:
                filtered_cache = filtered_cache[filtered_cache['team'] == team_name]
            return filtered_cache
            
        return pd.DataFrame()

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
    if passes.empty:
        return passes
        
    if 'location' not in passes.columns or 'pass_end_location' not in passes.columns:
        return passes
        
    passes = passes.dropna(subset=['location', 'pass_end_location'])
    if passes.empty:
        return passes
    
    # Parse string representations of lists if loaded from CSV cache
    import ast
    def parse_loc(loc):
        if isinstance(loc, str):
            try:
                return ast.literal_eval(loc)
            except:
                return [0.0, 0.0]
        return loc

    passes['location'] = passes['location'].apply(parse_loc)
    passes['pass_end_location'] = passes['pass_end_location'].apply(parse_loc)

    # Extract coordinates
    passes[['x', 'y']] = pd.DataFrame(passes['location'].tolist(), index=passes.index).iloc[:, :2]
    passes[['end_x', 'end_y']] = pd.DataFrame(passes['pass_end_location'].tolist(), index=passes.index).iloc[:, :2]
    
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

