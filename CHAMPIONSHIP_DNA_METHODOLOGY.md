# Championship DNA Comparison Methodology

This document outlines the mathematical logic utilized to calculate the **Tactical Evaluation Score (TES)**, a unified ranking system designed to quantify and rank top team tactical identities and structural dominance within a specific championship season.

## 1. Core Objectives
The primary goal of the Championship DNA comparison is to evaluate fundamentally robust tactical profiles by scoring teams on metrics like Expected Threat, passing coherence, vertical transition ability, expected goals (xG), and possession retention.

## 2. Passing Network Graph Construction & Metrics

All tactical identity structures are modeled using directed, weighted graphs derived directly from raw event data. 
Let the passing network be represented as a directed graph $$G = (V, E)$$, where:
- $$V$$ is the set of all unique players (nodes) who participated in a match.
- $E$ is the set of directed passing interactions (edges) between players.
- $w_{ij}$ represents the weight of the directed edge from player $i$ to player $j$, denoting the total volume of successful passes completed between them.

From this base graph structure, we extract several core structural attributes:

### Active Connected Volume (Passing Volume)
Passing volume signifies a team's sheer possession control and connective activity. It is mathematically the total sum of edge weights in the directed graph network:
$$\text{Volume} = \sum_{i \in V} \sum_{j \in V} w_{ij}$$

### Betweenness Centralization
Centralization measures the structural reliance a team places on specific individual "playmakers". We compute this by first calculating the Betweenness Centrality ($$C_B(v)$$) for every node, which quantifies the fraction of shortest paths that pass through that node.
Let $\sigma_{st}$ be the total number of shortest paths from node $s$ to node $t$, and $\sigma_{st}(v)$ be the number of those paths passing through 

$$C_B(v) = \sum_{s \neq v \neq t \in V} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Team Centralization is then calculated as the standard deviation ($\sigma$) of the betweenness centralities across all players in the network. A high Centralization indicates extreme reliance on a few star nodes, while low centralization dictates a decentralized, balanced passing architecture.

$$\text{Centralization}_{\text{team}} = \sqrt{\frac{1}{|V|} \sum_{v \in V} \left( C_B(v) - \mu_{C_B} \right)^2}$$

### Triadic Cohesion (Clustering)
Triadic cohesion quantifies the localized structural density of the team essentially, how effectively localized clusters of players pass the ball dynamically in triangles to support one another.
This is implemented by extracting the weighted clustering coefficient ($$C(v)$$) for every node. The team's overall Cohesion is the mean of these clustering values:

$$ Cohesion_{team} = \frac{1}{|V|} \sum_{v \in V} C(v) $$

Higher generalized cohesion represents strong, systemic short-range support structures universally embedded regardless of position.

## 3. Match Results & Win/Loss Ratios
While the TES operates independently of points entirely to act as an objective indicator of *how* a team plays, we still log their points trajectory to compare tactical identity against actual point accumulation:
Assuming a team plays $N$ matches, having $W$ wins, $D$ draws, and $L$ losses:
- **Win Ratio ($W_R$)**: $\displaystyle W_R = \frac{W}{N}$
- **Loss Ratio ($L_R$)**: $\displaystyle L_R = \frac{L}{N}$
- **Win-Loss Spread ($S_{WL}$)**: $\displaystyle S_{WL} = W_R - L_R$

The Win-Loss Spread naturally penalizes teams that lose frequently and rewards consistent winners, bound roughly between $[-1, 1]$. In practical terms, to avoid negative indices, we normalize $S_{WL}$ across the championship to a $$0 \rightarrow 1$$ scale:

$$ \hat{S}_{WL} = \frac{S_{WL} - \min(S_{WL})}{\max(S_{WL}) - \min(S_{WL})} $$

## 4. Tactical Efficacy Score (TES)
The Tactical Efficacy Score represents a single numerical value (0 to 1) outlining how "dominant" a team's playing style is. It aggregates our advanced "Team DNA" multi-dimensional network space into a single metric.

Before aggregation, each metric $M$ for every team is min-max normalized ($M_{norm}$) compared to the rest of the teams in the *same* championship:

$$ M_{norm_{i}} = \frac{M_i - \min(M_{all})}{\max(M_{all}) - \min(M_{all})} $$

### Components of the TES (8 Dimensions):
1. **Cohesion ($Coh$)**: Higher is better (dense triangle passing loops).
2. **Delta xT per Match ($TxT$)**: TransGoalNet Expected Threat generated per match.
3. **Basic xT per Match ($BxT$)**: Baseline positional expected threat per match.
4. **Decentralization ($Dec$)**: Calculated as $1 - \text{Centralization}_{norm}$. Rewards distributed network architectures rather than single playmakers.
5. **Expected Goals - xG ($xG$)**: Direct volume of traditional Expected Goals.
6. **Passing Accuracy ($PAcc$)**: Ratio of completed passes to intended passes.
7. **Retention Ability ($Ret$)**: The structural stability under possession sequences (Calculated via passes/duration per possession period).
8. **Verticality / Itrans ($ITr$)**: Transition efficiency modifier identifying quickly evolving threat, calculated as $\displaystyle I_{Trans} = \frac{\sum TransGoal\ xT}{Total\ Possession\ Duration}$.

**TES Calculation:**

$$ TES = (w_1 \overline{Coh}) + (w_2 \overline{TxT}) + (w_3 \overline{BxT}) + (w_4 \overline{Dec}) + (w_5 \overline{xG}) + (w_6 \overline{PAcc}) + (w_7 \overline{Ret}) + (w_8 \overline{ITr}) $$

Currently, the default heuristic weights assign an even 12.5% distribution across all 8 factors if MLR is deactivated:
- $w_1 = w_2 = w_3 = w_4 = w_5 = w_6 = w_7 = w_8 = 0.125$

*Note: The TES naturally scales from $0 \rightarrow 1$ assuming optimal components. An average team will sit around $0.4 - 0.6$.*

### Optimization Engine 1: Hybrid PCA-MLR
To solve the "Collinearity Problem" where tactical features are naturally correlated (e.g., high Cohesion often correlates with high Retention), we implement a **Hybrid PCA-MLR** approach.
1.  **Dimensionality Reduction (PCA):** We extract the first 3-4 Principal Components (orthogonal axes) that capture >80% of the tactical variance across all teams.
2.  **Orthogonal Regression:** We run an Ordinary Least Squares (OLS) regression using these uncorrelated PCs as inputs against the team Win Ratio.
3.  **Weight Back-Projection:** The resulting coefficients are projected back onto the original 8-dimensional space to derive stable, non-redundant feature importance weights.

### Optimization Engine 2: XGBoost & SHAP (Non-Linear)
For complex tactical landscapes where relationships are non-linear, we utilize an **XGBRegressor** (Gradient Boosting).
1.  **Non-Linear Modeling:** Captures decision-tree based relationships between metrics (e.g., Verticality is only positive if Passing Accuracy stays above a certain threshold).
2.  **SHAP (SHapley Additive exPlanations):** Instead of raw coefficients, we calculate the global average impact (mean absolute SHAP) of every feature on the model's predictions. These SHAP values determine the relative weights in the TES equation, ensuring the ranking is driven by actual predictive contribution.

## 4.2. Global Optimization & Validation
To establish a "Universal Tactical Benchmark," the framework supports **Global All-Time Optimization**. 
- **Cross-Competition Training:** The models aggregate match data across all available leagues and seasons (post a user-defined year threshold) to find universal "Winning DNA" traits.
- **Model Validation:** 
    - **80/20 Train/Test Split:** Models are trained on a subset of the global data and validated against held-out matches to ensure generalization.
    - **Early Stopping:** XGBoost training utilizes a validation set with a patience of 10 rounds to prevent "memorization" of specific team names and ensure tactical metrics are the primary drivers.
    - **R² Scoring:** The dashboard explicitly reports the $R^2$ coefficient of determination for both sets to monitor model health.

## 4.3. Unified Weight Pipeline
All optimized weights are serialized to a centralized `assets/` directory (`tes_pca_weights.json`, `tes_xgboost_weights.json`). The dashboard implements a **Hierarchical Fallback Mapping**:
1.  **Season-Specific Weights** (if trained)
2.  **League All-Time Weights** (if trained)
3.  **Global Historical Weights** (Master Fallback)

This ensures that the "Global" model provides a robust analytical baseline even for competitions with limited data.

## 5. Algorithmic Workflow
1. **Offline-First Data Resolution:** Check for local `{Team}_raw_events.csv` to bypass API latency. If missing, fetch match-level data via `statsbombpy` and cache locally.
2. **Global Result Aggregation:** Calculate Win/Loss ratios and Win-Loss spreads across the defined temporal window (e.g., matches since 2015).
3. **Multi-League DNA Retrieval:** Recursively load or compute DNA profiles across all relevant competitions to build the high-dimensional feature matrix.
4. **TES Model Inference:** Apply the chosen optimization engine (PCA-MLR or XGBoost) using weights loaded from the centralized `assets/` pipeline.
5. **Contextual Normalization:** Z-Score or Min-Max normalize the 8 DNA traits across the active dataset.
6. **Ranking:** Compute final **TES** and generate dynamic leaderboard visualizations.
