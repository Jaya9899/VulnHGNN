# VulnHGNN: Vulnerability Detection using Graph Neural Networks

VulnHGNN is a system designed to detect, localize, and repair software vulnerabilities using Graph Neural Networks (GNNs). It provides both a web-based UI for interactive analysis and a command-line interface (CLI) for automated pipeline execution. Additionally, it tracks carbon emissions associated with model execution.

## Features

- **Automated Vulnerability Detection**: Leverages PyTorch Geometric (PyG) and GNNs to analyze code for potential vulnerabilities.
- **Interactive Web UI**: A Flask-based frontend for visualizing, localizing, and repairing vulnerabilities.
- **End-to-End CLI Pipeline**: A robust command-line tool (`pipeline.py`) to process source files, detect issues, and generate reports.
- **Carbon Tracking**: Integrated with `codecarbon` to measure and report the environmental impact of your ML workloads.

## Prerequisites

- **Python 3.8+**
- Git (optional, for cloning)
- Windows OS (scripts provided are `.bat` files)

## Installation

1. **Clone the repository (or navigate to the project folder):**
   ```bash
   cd path\to\VulnerabilityDetectionGNN
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   ```bash
   # On Windows
   venv\Scripts\activate
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Web Interface (Recommended)

The easiest way to start the web UI is by using the provided launcher script. This will automatically activate the virtual environment and start the Flask server.

**Option A: Using the provided script (Windows)**
Simply double-click the `run.bat` file, or run it from the command line:
```bash
run.bat
```
The server will start at `http://127.0.0.1:5000`. Open this URL in your web browser.

**Option B: Manual start**
If you prefer to start it manually:
```bash
venv\Scripts\activate
cd src
python app.py
```

### 2. Command Line Interface (Pipeline)

To process a file directly through the CLI without the web interface, you can use the `run_pipeline.bat` script or run the Python script directly.

**Option A: Using the provided script (Windows)**
```bash
run_pipeline.bat
```
*Note: This script is pre-configured to run `test_files\test_02_cwe190_overflow.c` against the `results\pyg_gnn_v9\best_pyg_model.pt` model.*

**Option B: Manual execution with custom arguments**
To run the pipeline on a specific file of your choice:
```bash
venv\Scripts\activate
set PYTHONPATH=%PYTHONPATH%;%CD%\src
python src\pipeline.py test_files\your_test_file.c --no-carbon --model results\pyg_gnn_v9\best_pyg_model.pt
```

## Project Structure

- `src/` - Source code for the backend (Flask app, Pipeline, Models).
- `data/` & `dataset/` - Data directories for processing graphs and models.
- `results/` - Contains trained models (e.g., `best_pyg_model.pt`) and evaluation outputs.
- `test_files/` - Sample code files (C/C++) used to test vulnerability detection.
- `docs/` - Documentation.
- `requirements.txt` - Python package dependencies.
- `run.bat` - Quick start launcher for the web application.
- `run_pipeline.bat` - Quick start launcher for the CLI pipeline.
