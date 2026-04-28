import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import pandas as pd
import numpy as np
import scipy.stats

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

def plot_dna_radar(dna_metrics, save_path=None, cdi=None, leaderboard_df=None):
    """Plots a normalized Radar Chart for Team DNA."""
    if not dna_metrics:
        return
        
    categories = ['Cohesion', 'Decentralization', 'Retention', 'Pass Accuracy', 'xG', 'ITrans', 'Basic xT', 'TransxT']
    
    metric_mapping = {
        'Cohesion': ('avg_cohesion', 'Cohesion'),
        'Decentralization': ('avg_centralization', 'Centralization'), 
        'Retention': ('avg_retention', 'Retention'),
        'Pass Accuracy': ('avg_pass_acc', 'Pass_Acc'),
        'xG': ('avg_xg', 'xG'),
        'ITrans': ('avg_itrans', 'ITrans'),
        'Basic xT': ('avg_xt', 'Basic_xT'),
        'TransxT': ('avg_trans_xt', 'Trans_xT')
    }
    
    def get_norm(team_metrics, metric_name):
        dict_key, df_key = metric_mapping[metric_name]
        val = team_metrics.get(dict_key, 0.0)
        
        if leaderboard_df is not None and not leaderboard_df.empty and df_key in leaderboard_df.columns:
            league_vals = leaderboard_df[df_key].dropna()
            if not league_vals.empty:
                c_min = league_vals.min()
                c_max = league_vals.max()
                if c_max > c_min:
                    if metric_name == 'Decentralization':
                        return max(0.01, min(1.0 - ((val - c_min) / (c_max - c_min)), 1.0))
                    else:
                        return max(0.01, min((val - c_min) / (c_max - c_min), 1.0))
                        
        fallback_max = {
            'Cohesion': 0.2, 'Decentralization': 1.0, 'Retention': 1.0, 
            'Pass Accuracy': 1.0, 'xG': 3.0, 'ITrans': 0.02, 
            'Basic xT': 2.0, 'TransxT': 15.0
        }
        max_val = fallback_max[metric_name]
        if metric_name == 'Decentralization':
            return max(0.01, min((1.0 - val) / max_val, 1.0)) if val < 1.0 else 0.01
        return max(0.01, min(val / max_val, 1.0))

    values = [get_norm(dna_metrics, cat) for cat in categories]
    
    raw_values = []
    for cat in categories:
        dict_key, _ = metric_mapping[cat]
        val = dna_metrics.get(dict_key, 0.0)
        if cat == 'Decentralization':
            raw_values.append(1.0 - val if val < 1.0 else 0.0)
        else:
            raw_values.append(val)
    
    # Close the radar loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Draw plot
    ax.plot(angles, values, color='#00ff85', linewidth=2, zorder=3)
    ax.fill(angles, values, color='#00ff85', alpha=0.25, zorder=3)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, weight='bold', color='white')
    ax.tick_params(axis='x', pad=55)
    
    # Formatting
    ax.set_ylim(0, 1.2)
    ax.set_yticklabels([]) # Hide radial ticks
    ax.spines['polar'].set_color('#555555')
    ax.grid(color='#555555', linestyle='--', linewidth=0.5)
    
    # Add actual values as text outside the circle
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
            
        ax.text(angle, 1.30, val_text, size=9, color='#00ff85', ha=ha, va='center', weight='bold')
        
    # Display CDI or TES Ring if provided (now expected as TES)
    if cdi is not None:
        norm_tes = max(0.01, min(cdi / 100.0, 1.0))
        circle_angles = np.linspace(0, 2 * np.pi, 100)
        ax.plot(circle_angles, [norm_tes]*100, color='gold', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
        ax.text(np.pi/4, norm_tes + 0.08, f"TES: {cdi:.2f}", color='gold', size=10, weight='bold', ha='center', va='center', zorder=5)
        
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', transparent=True)
        
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

def plot_championship_leaderboard(leaderboard_df):
    """Renders the Championship DNA Leaderboard."""
    st.subheader("🏆 Championship DNA Leaderboard")
    
    if leaderboard_df.empty:
        st.warning("No data available to build the leaderboard.")
        return
        
    st.markdown("This leaderboard ranks teams based on their **Tactical Evaluation Score (TES)**, which quantifies tactical dominance (Team DNA Metrics) independently from results.")
    
    # Format the dataframe for display
    display_df = leaderboard_df.copy()
    
    # Sort
    display_df = display_df.sort_values(by='TES', ascending=False).reset_index(drop=True)
    display_df.index += 1 # 1-indexed ranks
    
    # Keep specific columns
    cols_to_show = ['Team', 'Matches', 'Win_Ratio', 'Loss_Ratio', 'TES']
    if 'Seasons_Saved' in display_df.columns:
        cols_to_show.insert(2, 'Seasons_Saved')
        
    if 'Cohesion' in display_df.columns:
        cols_to_show.extend(['Cohesion', 'Trans_xT', 'Basic_xT', 'Centralization', 'xG', 'Pass_Acc', 'Retention', 'ITrans'])
        
    display_df = display_df[cols_to_show]
    
    # Formatting
    format_dict = {
        'Win_Ratio': '{:.2%}',
        'Loss_Ratio': '{:.2%}',
        'TES': '{:.3f}',
        'Cohesion': '{:.3f}',
        'Trans_xT': '{:.3f}',
        'Basic_xT': '{:.3f}',
        'Centralization': '{:.3f}',
        'xG': '{:.2f}',
        'Pass_Acc': '{:.2%}',
        'Retention': '{:.1f}',
        'ITrans': '{:.4f}'
    }
    
    # Apply styling
    styled_df = display_df.style.format(format_dict).background_gradient(
        subset=['TES'], cmap='YlGn'
    ).background_gradient(
        subset=['Win_Ratio'], cmap='Greens'
    ).background_gradient(
        subset=['Loss_Ratio'], cmap='Reds'
    )
    
    st.dataframe(styled_df, use_container_width=True, height=600)

def plot_dual_dna_radar(team1_name, team1_metrics, team2_name, team2_metrics, leaderboard_df=None, tes1=None, tes2=None):
    """Plots a dual-team tactical radar chart normalized by min-max from leaderboard."""
    if not team1_metrics or not team2_metrics:
        st.warning("Insufficient data to plot dual radar.")
        return

    # 8 Spikes
    categories = ['Cohesion', 'Decentralization', 'Retention', 'Pass Accuracy', 'xG', 'ITrans', 'Basic xT', 'TransxT']
    
    # We need to map these to the keys in the metrics dict and leaderboard_df
    metric_mapping = {
        'Cohesion': ('avg_cohesion', 'Cohesion'),
        'Decentralization': ('avg_centralization', 'Centralization'), # special handling
        'Retention': ('avg_retention', 'Retention'),
        'Pass Accuracy': ('avg_pass_acc', 'Pass_Acc'),
        'xG': ('avg_xg', 'xG'),
        'ITrans': ('avg_itrans', 'ITrans'),
        'Basic xT': ('avg_xt', 'Basic_xT'),
        'TransxT': ('avg_trans_xt', 'Trans_xT')
    }
    
    def get_norm(team_metrics, metric_name):
        dict_key, df_key = metric_mapping[metric_name]
        val = team_metrics.get(dict_key, 0.0)
        
        if leaderboard_df is not None and not leaderboard_df.empty and df_key in leaderboard_df.columns:
            league_vals = leaderboard_df[df_key].dropna()
            if not league_vals.empty:
                c_min = league_vals.min()
                c_max = league_vals.max()
                if c_max > c_min:
                    if metric_name == 'Decentralization':
                        return max(0.01, min(1.0 - ((val - c_min) / (c_max - c_min)), 1.0))
                    else:
                        return max(0.01, min((val - c_min) / (c_max - c_min), 1.0))
                        
        fallback_max = {
            'Cohesion': 0.2, 'Decentralization': 1.0, 'Retention': 1.0, 
            'Pass Accuracy': 1.0, 'xG': 3.0, 'ITrans': 0.02, 
            'Basic xT': 2.0, 'TransxT': 15.0
        }
        max_val = fallback_max[metric_name]
        if metric_name == 'Decentralization':
            return max(0.01, min((1.0 - val) / max_val, 1.0)) if val < 1.0 else 0.01
        return max(0.01, min(val / max_val, 1.0))

    values1 = [get_norm(team1_metrics, cat) for cat in categories]
    values2 = [get_norm(team2_metrics, cat) for cat in categories]

    # Close the radar loop
    values1 += values1[:1]
    values2 += values2[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    # Team 1 (Deep Blue)
    color1 = '#1f77b4' # deep blue
    ax.plot(angles, values1, color=color1, linewidth=2, label=team1_name, zorder=3)
    ax.fill(angles, values1, color=color1, alpha=0.4, zorder=3)
    
    # Team 2 (Bright Orange)
    color2 = '#ff7f0e' # bright orange
    ax.plot(angles, values2, color=color2, linewidth=2, label=team2_name, zorder=4)
    ax.fill(angles, values2, color=color2, alpha=0.4, zorder=4)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    # Sans-serif font
    ax.set_xticklabels(categories, size=10, weight='bold', color='white', family='sans-serif')
    ax.tick_params(axis='x', pad=30)
    
    # Formatting
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='#888888', size=8)
    ax.spines['polar'].set_color('#555555')
    ax.grid(color='#555555', linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1b1b1b', edgecolor='#555555', labelcolor='white', prop={'family': 'sans-serif', 'weight': 'bold', 'size': 10})
    
    # Display CDI or TES Rings if provided
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    
    if tes1 is not None:
        norm_tes1 = max(0.01, min(tes1 / 100.0, 1.0))
        ax.plot(circle_angles, [norm_tes1]*100, color=color1, linestyle='--', linewidth=2, alpha=0.8, zorder=2)
        ax.text(np.pi/4, norm_tes1 + 0.05, f"TES(Blue): {tes1:.2f}", color='white', size=7, weight='bold', ha='center', va='center', zorder=5)
        
    if tes2 is not None:
        norm_tes2 = max(0.01, min(tes2 / 100.0, 1.0))
        ax.plot(circle_angles, [norm_tes2]*100, color=color2, linestyle='--', linewidth=2, alpha=0.8, zorder=2)
        ax.text(3*np.pi/4, norm_tes2 + 0.05, f"TES(Orange)): {tes2:.2f}", color='white', size=7, weight='bold', ha='center', va='center', zorder=5)
    
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
