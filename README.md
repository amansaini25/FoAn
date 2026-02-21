# ⚽ Vibe: Football Analytics Framework

**Vibe** is a specialized football analytics dashboard designed to analyze and visualize team tactical identity, specifically focusing on passing networks and **Expected Threat (xT)**. The current implementation uses data from the **Indian Super League (ISL) 2021/22 season** (via StatsBomb) to benchmark Hyderabad FC's championship-winning tactical connectivity.

This repository represents the modular, production-ready framework for the project, refactored from an initial monolithic proof-of-concept.

## 🌟 Key Features

### 1. 🕸️ Network Health Metrics
*   **Centralization (Std Dev):** Measures the team's reliance on specific players. High centralization indicates heavy dependence on key playmakers, while low centralization suggests a more distributed approach.
*   **Triadic Cohesion:** Evaluates the strength of local support triangles within the passing network, indicating how well players support each other in possession.
*   **Pass Volume:** Total number of successful passes in the analyzed period.
*   **Active Connections:** The number of unique passer-receiver pairs.

### 2. ⚡ Directed Passing Network
*   **Visualization:** A pitch map displaying player average positions and passing links.
*   **Weighted Arrows:** Thickness represents passing frequency between players.
*   **xT-Sized Nodes:** Node size corresponds to the player's total Expected Threat (xT) contribution, highlighting key threat creators.
*   **Interactive Filtering:** Adjust the minimum pass count to filter out less significant connections.

### 3. 📈 Advanced Analytics
*   **Expected Threat (xT):** Calculates the probability of a possession resulting in a goal based on pass origin and destination zones.
*   **Zone Activity (Heatmaps):** Kernel Density Estimation (KDE) plots showing where the team or specific players operate most frequently on the pitch.
*   **Threat Pulse:** A timeline analysis visualizing threat generation by top players across match phases (15-minute intervals).

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Network Science:** [NetworkX](https://networkx.org/)
*   **Visualization:** [Mplsoccer](https://mplsoccer.readthedocs.io/), [Matplotlib](https://matplotlib.org/)
*   **Data Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) via `statsbombpy`

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd football_analytics_framework
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Dashboard

To launch the application, run the following command from the project root:

```bash
streamlit run src/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## 📂 Project Structure

```
football_analytics_framework/
├── src/
│   ├── app.py              # Main application entry point
│   ├── components/         # UI components (Sidebar, Visuals, Layout)
│   ├── engine/             # Analytical logic (xT Model, Network Metrics)
│   └── utils/              # Helper functions (Data Loader, Styling)
├── assets/
│   └── style.css           # Custom CSS for Streamlit
├── data/                   # Data storage (Raw/Processed)
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 📊 Data Source
This project utilizes open data provided by **StatsBomb**.
*   **Competition:** Indian Super League (ISL) 2021/22
*   **Focus Team:** Hyderabad FC

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
