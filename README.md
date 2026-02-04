# Light Power QC Dashboard

A professional Streamlit dashboard for quality control and analysis of light source power measurements.

## Features

- **Analysis Dashboard**: Detailed per-dataset analysis with Linearity, Stability (Short/Long term) graphs.
- **Metrics History**: Aggregated historical view of key metrics (Stability, Linearity R²) across all datasets.

## Setup

This project uses `uv` for dependency management.

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

## Running the Dashboard

To start the Streamlit application:

```bash
uv run streamlit run main.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

## Project Structure

- `main.py`: Main Streamlit application entry point.
- `data_loader.py`: Data parsing and aggregation logic.
- `light_source_power_datasets/`: Directory containing dataset folders (CSV and YAML files).
