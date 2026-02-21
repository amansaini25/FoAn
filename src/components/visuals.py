import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import pandas as pd

def plot_passing_network(filtered_df, min_pass_count):
    """Plots the directed passing network."""
    st.subheader(f"🕸️ Directed Passing Network (Min Passes: {min_pass_count})")
    
    # Prepare Data
    avg_locs = filtered_df.groupby('player_name')[['x', 'y']].mean()
    pass_counts = filtered_df.groupby(['player_name', 'pass_recipient_name']).size().reset_index(name='weight')
    
    # Pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1b1b1b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(10, 7))
    
    # Draw Edges (Arrows)
    strong_links = pass_counts[pass_counts['weight'] >= min_pass_count]
    for _, row in strong_links.iterrows():
        p1, p2 = row['player_name'], row['pass_recipient_name']
        if p1 in avg_locs.index and p2 in avg_locs.index:
            alpha = min(1, row['weight'] / 15)
            pitch.arrows(avg_locs.loc[p1].x, avg_locs.loc[p1].y,
                         avg_locs.loc[p2].x, avg_locs.loc[p2].y,
                         ax=ax, width=2, headwidth=4, color='#00ff85', alpha=alpha, zorder=1)

    # Draw Nodes (Sized by xT)
    xt_sum = filtered_df.groupby('player_name')['xT'].sum()
    for player, loc in avg_locs.iterrows():
        sz = xt_sum.get(player, 0) * 5000
        pitch.scatter(loc.x, loc.y, ax=ax, s=max(sz, 100), color='#ff4b4b', edgecolors='white', zorder=2)
        ax.text(loc.x, loc.y+3, player.split()[-1], color='white', fontsize=9, ha='center', zorder=3)
        
    st.pyplot(fig)

def plot_top_xt(filtered_df):
    """Displays top xT generators."""
    st.subheader("⚡ Critical Nodes (Top xT)")
    leaderboard = filtered_df.groupby('player_name')[['xT']].sum().sort_values('xT', ascending=False).head(5)
    st.dataframe(leaderboard.style.background_gradient(cmap='Reds'))

def plot_zone_activity(filtered_df):
    """Plots KDE heatmap of pass locations."""
    st.subheader("🎯 Zone Activity")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    pitch2 = VerticalPitch(pitch_type='statsbomb', line_zorder=2, line_color='#c7d5cc')
    pitch2.draw(ax=ax2)
    if not filtered_df.empty:
        pitch2.kdeplot(filtered_df.x, filtered_df.y, ax=ax2, cmap='magma', fill=True, levels=10, alpha=0.5)
    st.pyplot(fig2)

def plot_threat_pulse(pass_df, filtered_df):
    """Plots threat generation over time."""
    st.markdown("---")
    st.subheader("📈 The Pulse: Threat Generation Over Time")

    if filtered_df.empty:
        st.write("No data available for the selected filters.")
        return

    # Top 5 players by total xT from filtered data
    top_players = filtered_df.groupby('player_name')['xT'].sum().nlargest(5).index
    timeline_df = pass_df[pass_df['player_name'].isin(top_players)]

    if not timeline_df.empty:
        # Ensure time_bin is present
        bins = [0, 15, 30, 45, 60, 75, 90, 120]
        labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
        timeline_df['time_bin'] = pd.cut(timeline_df['minute'], bins=bins, labels=labels, right=False)
        
        pivot_timeline = timeline_df.groupby(['time_bin', 'player_name'])['xT'].sum().unstack().fillna(0)
        st.line_chart(pivot_timeline)
        st.caption("Which players generate the most threat during specific 15-minute match phases?")
