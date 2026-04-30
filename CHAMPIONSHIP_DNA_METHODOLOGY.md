# Tactical Efficiency Score (TES) & DNA Methodology: A Structural Performance Framework

## 1. Abstract
This document formalizes the mathematical and statistical methodologies employed in the **Tactical Efficiency Score (TES)** framework. The framework transitions from traditional point-based metrics to a structural identity model that evaluates team performance through the lens of graph theory, spatial threat, and defensive suppression. By utilizing **Match-Level DNA** feature extraction and **Match-Goal Difference ($\Delta G$)** as the primary regressor, the model identifies the fundamental tactical components that drive consistent competitive dominance.

---

## 2. Structural Identity: Passing Network Graph Construction
The foundation of the DNA model is the representation of a football match as a directed, weighted graph $G = (V, E)$, where $V$ is the set of nodes (players) and $E$ is the set of directed edges (successful passes).

### 2.1. Betweenness Centralization ($\sigma_{C_B}$)
Centralization quantifies the structural fragility or "playmaker-dependency" of a network. We define the team-level centralization as the standard deviation of node-level betweenness centralities ($C_B(v)$), capturing the variance in the distribution of shortest-path control.
$$C_B(v) = \sum_{s \neq v \neq t \in V} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
$$\text{Centralization}_{\text{team}} = \sqrt{\frac{1}{|V|} \sum_{v \in V} (C_B(v) - \mu_{C_B})^2}$$

### 2.2. Triadic Cohesion (Clustering)
To evaluate localized structural density—the ability to form dynamic passing triangles—we utilize the mean weighted clustering coefficient. High cohesion indicates a resilient, multi-path connection architecture.
$$\text{Cohesion}_{\text{team}} = \frac{1}{|V|} \sum_{v \in V} C(v)$$

---

## 3. Advanced Tactical Metrics
The DNA profile consists of 10 dimensions, integrating the following state-of-the-art metrics:

### 3.1. Defensive Suppression Index (DSI)
DSI evaluates a team's ability to force sub-optimal opponent behavior through pressure. It incorporates spatial weighting ($w_s = 1.5$ in high-value zones, $1.0$ elsewhere) and scales by the opponent's resulting failure rate.
$$DSI = \frac{\sum_{p \in P_{opp}} (w_s \cdot \mathbb{I}_{\text{pressure}}(p))}{|P_{opp}|} \times (1 - \text{Pass Completion Rate}_{opp})$$

### 3.2. High-Efficiency Suppression (HES)
HES measures the conversion of pressure into immediate possession retrieval. It is a temporal correlation metric identifying "three-second recovery windows" following a pressure event.
$$HES = \sum_{p \in P_{\text{team}}} \mathbb{I}(\exists r \in R_{\text{team}} : t_r \in [t_p, t_p + 3s])$$
*Where $R$ is the set of Interceptions and Ball Recoveries.*

### 3.3. TransGoalNet & Delta xT ($\Delta xT$)
Utilizing a Graph Transformer ($k=5$ action window), we compute the dynamic Expected Threat surplus ($\Delta xT$) generated during transition phases, identifying verticality and structural progression.

---

## 4. Modeling & Optimization Strategy
The TES engine employs a dual-stage optimization process to derive the weighted composite score.

### 4.1. Transition to Match-Level Modeling
To increase sample size and capture tactical variance across different contexts, the target variable has transitioned from season-level win ratios to **Match-Level Goal Difference ($\Delta G$)**. This transforms the problem into a regression task where we predict the scoring margin based on the match-day DNA profile.

### 4.2. Feature Engineering & Intra-Group Standardization
To isolate tactical efficiency from differing league qualities (e.g., comparing the Premier League to the ISL), features are **Z-score normalized within each competition and season**. This ensures the models learn "Competitive Advantage" relative to the specific environment rather than raw volume metrics.
$$\hat{x}_{i,c,s} = \frac{x_{i,c,s} - \mu_{c,s}}{\sigma_{c,s}}$$

### 4.3. Optimization Engines
*   **Engine 1: Hybrid PCA-MLR**: Utilizes Principal Component Analysis to extract orthogonal tactical components, followed by OLS regression. This eliminates multi-collinearity between metrics like Cohesion and Retention.
*   **Engine 2: XGBoost (Gain-based)**: A non-linear gradient-boosted tree model. We extract feature importance via the **Gain metric** (average reduction in loss brought by a feature), providing a precise structural importance weight for each DNA dimension.

### 4.4. Validation & Generalization (GroupKFold)
To prevent the model from "memorizing" specific high-performing team names (Team Identity Leakage), we implement **GroupKFold Cross-Validation**.
*   **Grouping Criterion**: `team_id`
*   **Mechanism**: The model is trained on one set of teams and validated on a completely disjoint set of teams. This forces the model to learn the **Tactical Laws** of success that generalize across all entities.

---

## 5. The Tactical Efficiency Score (TES) Formula
The final TES is a weighted summation of 10 normalized dimensions ($D_1 \dots D_{10}$), providing a unified index of structural dominance:

$$TES = \sum_{j=1}^{10} w_j \cdot D_{j, \text{norm}}$$

**Feature Dimensions:**
1.  **Cohesion** ($w_{coh}$)
2.  **Trans_xT** ($w_{txt}$)
3.  **Basic_xT** ($w_{bxt}$)
4.  **Decentralization** ($w_{dec}$)
5.  **Expected Goals (xG)** ($w_{xg}$)
6.  **Passing Accuracy** ($w_{pacc}$)
7.  **Retention** ($w_{ret}$)
8.  **Verticality (ITrans)** ($w_{itr}$)
9.  **Defensive Suppression (DSI)** ($w_{dsi}$)
10. **High-Efficiency Suppression (HES)** ($w_{hes}$)

---

## 6. Algorithmic Workflow
1.  **ETL Pipeline**: Resolve match-level event streams from StatsBomb API or local caches.
2.  **DNA Extraction**: Compute 10 structural metrics per team per match.
3.  **Cross-Competition Standardization**: Group by (League, Season) and apply Z-score scaling.
4.  **Batch Training**: Execute XGBoost/PCA-MLR on the global dataset using GroupKFold.
5.  **Weight Persistence**: Export `optimized_dna_weights.json` to the unified `assets/` directory.
6.  **Inference**: Apply global weights to any selected team profile to compute the real-time TES ranking.
