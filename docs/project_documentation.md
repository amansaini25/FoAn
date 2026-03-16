# FoAn Football Analytics: Comprehensive Project Documentation

## 1. Executive Summary
FoAn represents a targeted evolution in football analytics, pivoting away from traditional counting stats (possession %, raw pass counts) toward systemic valuation. By harmonizing network science with Expected Threat (xT), the platform quantifies the geometry of team play, highlighting not just who passes the ball, but who advances the underlying probability of a goal. 

## 2. Architectural Blueprint
The system follows a modular architecture to separate data extraction, analytical processing, and UI rendering:

### Data Ingestion Layer (`src/utils/data_loader.py`)
Responsible for communicating with the StatsBomb API via `statsbombpy`. Key optimizations include:
- Aggressive `@st.cache_data` decorators to minimize redundant API calls.
- Automated coordinate extraction and data sanitization routines bridging raw StatsBomb events into functional actions.

### Analytical Engine (`src/engine/`)
- **Network Metrics (`metrics.py`):** Utilizes `NetworkX` to assemble directed passing graphs. Key measures include standard deviation of node centrality (which uncovers centralized vs. distributed playstyles) and triadic cohesion.
- **Expected Threat Model (`xt_model.py`):** Translates events into spatial values mapping an 8x12 grid over the pitch. The model leverages dynamic programming to recursively calculate transition matrices (moves & shots) mapping every location's goal probability. Included is a robust checkpointing system (`save_checkpoint`/`load_checkpoint`) ensuring zero-latency inference after an initial full dataset training.

### Application & Component Layer (`src/app.py` & `src/components/`)
- **Main App:** Orchestrates the UI via a multi-tab view organizing "Network Identity" analysis alongside an "xT Evaluation Grid".
- **Visuals (`visuals.py`):** Heavily relies on `mplsoccer` for bespoke football pitch rendering. Supports layered scatter plots, network edges with dynamic alpha/width corresponding to volume, KDE heatmaps for positional tendences, and raw xT arrays mapping spatial grid valuation.
- **Sidebar (`sidebar.py`):** Provides complex query mechanisms (Championship -> Season -> Team -> Venue -> Phase) allowing for detailed sub-filtering of matches and tracking.

## 3. Product Features Detailed Breakdown
- **Network Centralization & Cohesion:** Distinguishes between "star-driven" passing structures and highly decentralized, resilient passing networks that maintain strong local supporting triangles.
- **xT Grid Visualizer:** Peels back the curtain on the AI model by projecting a heatmap of the base Markov probabilities across a full football pitch.
- **Threat Pulse Strategy:** Evaluates stamina and tactical endurance by slicing Expected Threat generation linearly across the match time-bins (0-15', 15-30', etc.). Ensures analysts can spot late-game drop-offs or systemic flaws during specific match phases.

## 4. Future Roadmap & Iterations
1. **Defensive Action Networks:** Integrate defensive pressures and interceptions into a secondary network.
2. **Dynamic UI Expansion:** Enhance interactivity of the Expected Threat grid to allow users to click paths and calculate sequence values natively.
3. **Database Caching:** Hook the Statsbomb layer into a persistent SQLite/Postgres layer to deprecate memory caching for significantly faster start-ups independent of the API constraint.
