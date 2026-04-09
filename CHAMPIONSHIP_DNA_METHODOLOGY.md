# Championship DNA Comparison Methodology

This document outlines the mathematical logic utilized to calculate the **Championship DNA Index (CDI)**, a unified ranking system designed to compare top team tactical identities against their actual on-field success (Winning/Losing ratios) within a specific championship season.

## 1. Core Objectives
The primary goal of the Championship DNA comparison is to evaluate if playing an aesthetically and structurally dominant style of football ("good DNA") functionally translates to winning games. We merge advanced network metrics (Expected Threat, passing coherence) with classic match outcome statistics.

## 2. Match Results & Win/Loss Ratios
For a selected team in a specific competition and season, their success is evaluated purely on their points trajectory.
Assuming a team plays $N$ matches, having $W$ wins, $D$ draws, and $L$ losses:

- **Win Ratio ($W_R$)**: $\displaystyle W_R = \frac{W}{N}$
- **Loss Ratio ($L_R$)**: $\displaystyle L_R = \frac{L}{N}$
- **Win-Loss Spread ($S_{WL}$)**: $\displaystyle S_{WL} = W_R - L_R$

The Win-Loss Spread naturally penalizes teams that lose frequently and rewards consistent winners, bound roughly between $[-1, 1]$. In practical terms, to avoid negative indices, we normalize $S_{WL}$ across the championship to a $0 \rightarrow 1$ scale:

$$ \hat{S}_{WL} = \frac{S_{WL} - \min(S_{WL})}{\max(S_{WL}) - \min(S_{WL})} $$

## 3. Tactical Efficacy Score (TES)
The Tactical Efficacy Score represents a single numerical value outlining how "dominant" a team's passing network is. It aggregates our standard "Team DNA" metrics: `avg_cohesion`, `avg_xt`, `avg_trans_xt`, and `avg_centralization`.

Before aggregation, each metric $M$ for every team is min-max normalized ($M_{norm}$) compared to the rest of the teams in the *same* championship:

$$ M_{norm_{i}} = \frac{M_i - \min(M_{all})}{\max(M_{all}) - \min(M_{all})} $$

### Components of the TES:
1. **Cohesion ($Coh$)**: Higher is better (dense triangle passing loops). Weight $w_1 = 0.25$.
2. **Delta xT per Match ($TxT$)**: TransGoalNet Expected Threat generated per match. Weight $w_2 = 0.35$.
3. **Basic xT per Match ($BxT$)**: Baseline positional expected threat per match. Weight $w_3 = 0.20$.
4. **Decentralization ($Dec$)**: $1 - \text{Centralization}_{norm}$. We reward teams whose network load is distributed across the pitch rather than relying on a single playmaker. Weight $w_4 = 0.20$.

**TES Calculation:**

$$ TES = (w_1 \times \overline{Coh}) + (w_2 \times \overline{TxT}) + (w_3 \times \overline{BxT}) + (w_4 \times \overline{Dec}) $$

Currently, the default heuristic weights are distributed as:
- $w_1 = 0.25$, $w_2 = 0.35$, $w_3 = 0.20$, $w_4 = 0.20$

*Note: The TES naturally scales from $0 \rightarrow 1$ assuming optimal components. An average team will sit around $0.4 - 0.6$.*

## 3.1. Advanced TES Weighting Metrics (Mathematical Optimization)
While heuristic (expert-assigned) components offer baseline accuracy, findings natively suggest that the exact relationship governing elite play varies across competitions. To optimize the TES calculation without subjective bias, our framework dynamically applies robust statistical modeling natively within the metric algorithms.

### Implemented: Multiple Linear Regression (MLR)
The precise influence of each tactical parameter is mapped directly to actual $Win\_Ratio$ through an enforced-positive Ordinary Least Squares regression executed via Scikit-Learn. The application UI allows users to directly push a button to train an MLR model targeting the specific season (or an aggregated all-time history) live.
- **Target Variable (`y`)**: Actual Championship Win Ratio ($W_R$)
- **Input Features (`X`)**: Normalized parameter tensor $\rightarrow [Coh_{norm}, TxT_{norm}, BxT_{norm}, Dec_{norm}]$

The isolated, normalized coefficients ($\beta_1, \beta_2, \beta_3, \beta_4$) extracted from the converged linear equation seamlessly translate into perfectly optimal, data-driven model structural weights ($w_1, w_2, w_3, w_4$) that perfectly represent that specific league's tactical meta. These exact floats are generated, securely cached to the local file system (`tes_mlr_weights.json`), and automatically parsed on render to construct the dynamic Championship Leaderboards.

*(Note: In the isolated event where the MLR strictly converges onto negative correlations—mathematically invalidating the construct integrity of an additive score where zero represents absolute failure—the algorithm safely catches this and gracefully regresses to the generalized heuristic weights defined above).*

### Future Enhancement: Principal Component Analysis (PCA)
If teams exhibit heavily correlated behaviors (e.g., high $Coh$ almost always results in high $BxT$), calculating the Eigen-Vectors across the DNA Dataset will produce the "Principal Axis of Play" describing the ultimate variance in structural identity. Weighting metrics by their percentage contribution to $PC1$ constructs a fully data-driven equation for absolute network efficiency.

## 4. Championship DNA Index (CDI)
The ultimate ranking metric merges tactical dominance with pure winning effectiveness. 

$$ CDI = (TES \times \hat{S}_{WL}) \cdot 100 $$

### Interpretation of CDI
- **High CDI (>75)**: "The Blueprint". A team that combines dominant, expansive football (High TES) with ruthless efficiency in gathering points (High Spread). e.g., Championship-winning teams.
- **Moderate CDI (40-75)**: Good tactical blueprint but potentially lacking conversion into points, or an ugly-but-effective counter-attacking team that gathers points despite low passing network dominance.
- **Low CDI (<40)**: Structurally disjointed teams suffering on both tactical output and match outcomes.

## 5. Algorithmic Workflow
1. Fetch all match-level data for all teams in the chosen Competition & Season using `statsbombpy`.
2. Extract the Win/Loss results per team to calculate Spread ($S_{WL}$).
3. Load previously cached DNA Profiles (`dna_profile.json`) or rapidly compute passing metrics via grouped arrays to evaluate $TES$.
4. Z-Score or Min-Max normalize both $S_{WL}$ and the DNA traits.
5. Compute the final **CDI** and rank teams descending to build the Leaderboard.
