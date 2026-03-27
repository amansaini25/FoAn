import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import pandas as pd
import numpy as np

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
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    pitch2 = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='#c7d5cc', pitch_color='#1b1b1b')
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

def plot_xt_grid(xt_model):
    """Plots the xT grid as a heatmap."""
    st.subheader("🗺️ Expected Threat (xT) Evaluation Grid")
    
    if xt_model is None or xt_model.xT is None:
        st.warning("xT model not available.")
        return
        
    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1b1b1b', line_color='#c7d5cc')
    pitch.draw(ax=ax)
    
    # xT array is of shape (w=8, l=12). Statsbomb pitch is 120x80.
    im = ax.imshow(xt_model.xT, extent=[0, 120, 80, 0], cmap='viridis', alpha=0.6, aspect='auto')
    
    w, l = xt_model.xT.shape
    x_bins = np.linspace(0, 120, l + 1)
    y_bins = np.linspace(0, 80, w + 1)
    
    for i in range(w):
        for j in range(l):
            val = xt_model.xT[i, j]
            cx = (x_bins[j] + x_bins[j+1]) / 2
            cy = (y_bins[i] + y_bins[i+1]) / 2
            ax.text(cx, cy, f"{val:.3f}", color='white', ha='center', va='center', fontsize=8)
            
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Expected Threat (xT)")
    st.pyplot(fig)

def plot_dna_radar(dna_metrics):
    """Plots a normalized Radar Chart for Team DNA."""
    if not dna_metrics:
        return
        
    categories = ['Pass Vol', 'Centralization', 'Active Conns', 'Triadic Cohesion', 'xT', 'Delta xT']
    
    # Extract values
    raw_values = [
        dna_metrics.get('avg_pass_volume', 0.0),
        dna_metrics.get('avg_centralization', 0.0),
        dna_metrics.get('avg_active_connections', 0.0),
        dna_metrics.get('avg_cohesion', 0.0),
        dna_metrics.get('avg_xt', 0.0),
        dna_metrics.get('delta_xt', 0.0)
    ]
    
    # Cap values for normalization (empirical maximums to scale the polygon)
    max_vals = [1000, 0.2, 350, 0.2, 2.0, 1.0]
    
    # Normalize values between 0 and 1
    values = [max(0.01, min(v / m, 1.0)) for v, m in zip(raw_values, max_vals)]
    
    # Close the radar loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Draw plot
    ax.plot(angles, values, color='#00ff85', linewidth=2)
    ax.fill(angles, values, color='#00ff85', alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, weight='bold', color='white')
    
    # Formatting
    ax.set_ylim(0, 1)
    ax.set_yticklabels([]) # Hide radial ticks
    ax.spines['polar'].set_color('#555555')
    ax.grid(color='#555555', linestyle='--', linewidth=0.5)
    
    # Add actual values as text
    for angle, value, raw_value in zip(angles[:-1], values[:-1], raw_values):
        ha = 'center'
        if 0.1 < angle < np.pi - 0.1:       # Right half
            ha = 'left'
        elif np.pi + 0.1 < angle < 2*np.pi - 0.1: # Left half
            ha = 'right'
        
        # Display formatted raw value
        if raw_value >= 10:
            val_text = f" {int(raw_value)} "
        else:
            val_text = f" {raw_value:.2f} "
            
        ax.text(angle, value + 0.15, val_text, size=10, color='#ff4b4b', ha=ha, va='center', weight='bold')
        
    st.pyplot(fig)

def plot_tactical_heatmap(filtered_df, top_lane):
    """Plots Tactical Heatmap showing the Key Passing Lane based on TransGoalNet Attention."""
    import streamlit as st
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    
    st.subheader("🔥 Tactical Heatmap & Key Passing Lane")
    
    if filtered_df.empty:
        st.warning("No data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='#c7d5cc', pitch_color='#1b1b1b')
    pitch.draw(ax=ax)
    
    # Base Heatmap
    pitch.kdeplot(filtered_df.x, filtered_df.y, ax=ax, cmap='magma', fill=True, levels=10, alpha=0.3)
    
    # Key Passing Lane
    if top_lane and top_lane.get('passer') and top_lane.get('recipient'):
        p1 = top_lane['passer']
        p2 = top_lane['recipient']
        val = top_lane['attention']
        
        # Get average locs
        locs = filtered_df.groupby('player_name')[['x', 'y']].mean()
        
        if p1 in locs.index and p2 in locs.index:
            x1, y1 = locs.loc[p1].x, locs.loc[p1].y
            x2, y2 = locs.loc[p2].x, locs.loc[p2].y
            
            # Draw glowing arrow
            pitch.arrows(x1, y1, x2, y2, ax=ax, width=5, headwidth=8, color='#00ff85', alpha=0.9, zorder=3)
            
            # Nodes
            pitch.scatter(x1, y1, ax=ax, s=300, color='#ff4b4b', edgecolors='white', zorder=4)
            pitch.scatter(x2, y2, ax=ax, s=300, color='#ff4b4b', edgecolors='white', zorder=4)
            
            # Labels
            ax.text(x1, y1+3, p1.split()[-1], color='white', fontsize=11, ha='center', weight='bold', zorder=5)
            ax.text(x2, y2+3, p2.split()[-1], color='white', fontsize=11, ha='center', weight='bold', zorder=5)
            
            # Annotation
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y - 2, f"Attention: {val:.3f}", color='#00ff85', fontsize=12, ha='center', weight='bold', zorder=5)
            
            st.success(f"**Key Focus:** The model optimally routed attacks through the '{p1} ➡️ {p2}' channel (Attention: **{val:.3f}**).")
        else:
            st.info(f"Target node '{p1}' or '{p2}' not found in the current timeframe/filter.")
    else:
        st.info("No Key Passing Lane detected in the current data.")

    st.pyplot(fig)

