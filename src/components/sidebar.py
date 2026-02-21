import streamlit as st
import pandas as pd

def render_sidebar(pass_df):
    """
    Renders the sidebar controls and returns the filtered dataframe and analysis parameters.
    """
    st.sidebar.header("🔍 Analysis Controls")

    # A. Data Filters
    outcomes = ['All'] + list(pass_df['outcome_result'].unique()) if not pass_df.empty else ['All']
    selected_outcome = st.sidebar.selectbox("Match Outcome", outcomes)

    # B. Time Bins
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
    
    if not pass_df.empty:
        pass_df['time_bin'] = pd.cut(pass_df['minute'], bins=bins, labels=labels, right=False)
    
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
    filtered_df = pass_df.copy()
    if not filtered_df.empty:
        if selected_outcome != 'All':
            filtered_df = filtered_df[filtered_df['outcome_result'] == selected_outcome]
        if selected_time != 'Full Match':
            filtered_df = filtered_df[filtered_df['time_bin'] == selected_time]

    st.sidebar.info(f"Analyzing {len(filtered_df)} passes.")
    
    return filtered_df, min_pass_count, selected_time
