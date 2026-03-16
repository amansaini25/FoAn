# FoAn: Football Analytics Framework

**FoAn** is a modular, scalable football analytics dashboard designed to analyze and visualize team tactical identity. It specifically focuses on passing networks, triadic cohesion, and **Expected Threat (xT)**. This project leverages open data from StatsBomb to benchmark tactical connectivity, with an emphasis on championship-winning structures (like Hyderabad FC's 2021/22 ISL season).

## Core Technologies
- **UI/Frontend:** [Streamlit](https://streamlit.io/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Network Science:** [NetworkX](https://networkx.org/) (Centrality, Cohesion)
- **Visualization:** [Mplsoccer](https://mplsoccer.readthedocs.io/), [Matplotlib](https://matplotlib.org/)
- **Data Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) via `statsbombpy`

## Project Structure
- `src/app.py`: The main functional entry point for the Streamlit dashboard.
- `src/engine/`: Analytical core containing the Expected Threat (xT) model and network metrics.
- `src/components/`: Modular UI components including the sidebar and key visualizers.
- `src/utils/`: Helpers for data loading, logging, and styling.
- `assets/`: Custom CSS and static artifacts like pre-trained model weights.
- `docs/`: Comprehensive project documentation.

## Key Features
1. **Network Health Metrics:** Evaluates team reliance on specific players (Centralization) and local support strengths (Triadic Cohesion).
2. **Directed Passing Network:** Visualizes strong tactical links and critical nodes powered by Expected Threat generation.
3. **Expected Threat (xT) AI Model:** Employs a Markov-based dynamic programming approach to calculate the incremental probability of scoring from passing sequences. It handles full dataset training and caches the learned matrices natively for zero-latency inference.
4. **xT Evaluation Grid:** An interactive pitch heatmap grading the underlying spatial threat generation logic.
5. **The Threat Pulse:** A timeline breaking down critical threat builders through 15-minute match phases.

## Installation & Setup

### Prerequisites
- Python 3.8+
- Recommended environment: `football` (Conda)

### Installation
1. Clone the repository and navigate to the project directory.
2. Install the necessary dependencies:
```bash
pip install streamlit pandas numpy networkx matplotlib statsbombpy mplsoccer
```

### Running the Dashboard
To start the modular dashboard:
```bash
cd football_analytics_framework
streamlit run src/app.py
```
