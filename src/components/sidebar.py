import streamlit as st
import pandas as pd
from utils.data_loader import get_competitions, get_matches

def render_data_selection():
    """Renders the data selection controls and returns the selected team and matches."""
    st.sidebar.header("📁 Data Selection")
    
    # Fetch competitions
    comps = get_competitions()
    if comps.empty:
        st.sidebar.error("Failed to load competitions.")
        return None, pd.DataFrame(), None
        
    # 1. Competition Selection
    competition_names = sorted(comps['competition_name'].unique().tolist())
    selected_comp_name = st.sidebar.selectbox("Competition", competition_names)
    
    # 2. Season Selection
    comp_data = comps[comps['competition_name'] == selected_comp_name]
    season_names = sorted(comp_data['season_name'].unique().tolist(), reverse=True)
    selected_season_name = st.sidebar.selectbox("Season", season_names)
    
    # Get IDs for the selected comp & season
    selected_row = comp_data[comp_data['season_name'] == selected_season_name].iloc[0]
    comp_id = selected_row['competition_id']
    season_id = selected_row['season_id']
    
    # Fetch Matches
    matches = get_matches(comp_id, season_id)
    if matches.empty:
        st.sidebar.warning("No matches found for this competition and season.")
        return selected_comp_name, selected_season_name, None, pd.DataFrame()
        
    # 3. Team Selection
    teams = pd.concat([matches['home_team'], matches['away_team']]).unique()
    teams.sort()
    selected_team = st.sidebar.selectbox("Team", teams)
    
    # 4. Match Venue Filter
    venue_filter = st.sidebar.radio("Match Venue", ["All", "Home", "Away"])
    
    # Filter matches based on team and venue
    if venue_filter == "All":
        team_matches = matches[(matches['home_team'] == selected_team) | (matches['away_team'] == selected_team)]
    elif venue_filter == "Home":
        team_matches = matches[matches['home_team'] == selected_team]
    else:
        team_matches = matches[matches['away_team'] == selected_team]
        
    return selected_comp_name, selected_season_name, selected_team, team_matches


def render_analysis_controls(pass_df):
    """Renders the analysis controls and returns the filtered dataframe and analysis parameters."""
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Analysis Controls")

    if pass_df is None or pass_df.empty:
        st.sidebar.warning("No pass data available to filter.")
        return pd.DataFrame(), 3, 'Full Match'

    # A. Data Filters
    outcomes = ['All'] + list(pass_df['outcome_result'].unique())
    selected_outcome = st.sidebar.selectbox("Match Outcome", outcomes)

    # B. Time Bins
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
    
    pass_df_copy = pass_df.copy()
    pass_df_copy['time_bin'] = pd.cut(pass_df_copy['minute'], bins=bins, labels=labels, right=False)
    selected_time = st.sidebar.selectbox("Time Phase", ['Full Match'] + labels)

    # C. Network Slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕸️ Network Density")
    min_pass_count = st.sidebar.slider(
        "Min Passes to Connect", 
        min_value=1, max_value=15, value=3, 
        help="Filter weak links to see the core structure."
    )

    # Apply Filters
    filtered_df = pass_df_copy.copy()
    if selected_outcome != 'All':
        filtered_df = filtered_df[filtered_df['outcome_result'] == selected_outcome]
    if selected_time != 'Full Match':
        filtered_df = filtered_df[filtered_df['time_bin'] == selected_time]

    st.sidebar.info(f"Analyzing {len(filtered_df)} passes.")
    
    return filtered_df, min_pass_count, selected_time
