# VulnHGNN

VulnHGNN is a tool we built to help detect, localize, and fix software vulnerabilities using Graph Neural Networks (GNNs). It comes with a web UI for when you want an interactive visual experience, as well as a command-line interface (CLI) for running automated pipelines. As a bonus, it also tracks the carbon emissions from running the models.

## What it does
- Detects vulnerabilities automatically using PyTorch Geometric and GNNs.
- Supports multi-label classification, meaning it can detect multiple types of vulnerabilities at once.
- Localizes exactly where the vulnerability is in your code.
- Features self-healing capabilities to automatically suggest and apply fixes.
- Provides an interactive web UI built with Flask to easily visualize and fix issues.
- Includes a full CLI pipeline to process source files and generate reports from your terminal.
- Tracks carbon emissions using CodeCarbon to measure the environmental impact of your ML workloads.

## What you need
Make sure you have Python 3.8 or newer installed. You'll also need Windows since we use `.bat` scripts for the launchers.

## How to set it up
1. Clone or download this repository to your machine.
2. Open your terminal or command prompt in the project folder.
3. Create a virtual environment by running `python -m venv venv`.
4. Activate the environment with `venv\Scripts\activate`.
5. Install everything you need by running `pip install -r requirements.txt`.

## Getting started

### Running the Web UI
The easiest way to use VulnHGNN is through the web app. Just double-click the `run.bat` file in the project folder, or run `run.bat` from your command prompt. It will automatically activate your environment and start the Flask server. Once it's running, open your browser and go to http://127.0.0.1:5000.

If you prefer doing it manually, you can activate your environment, navigate to the `src` folder, and run `python app.py`.

### Running the CLI
If you want to process files directly from the command line, you can use the `run_pipeline.bat` script. By default, it runs a test file against our pre-trained model. 

To run it on your own files, you can manually run the pipeline like this:
1. Activate your virtual environment.
2. Set your python path: `set PYTHONPATH=%PYTHONPATH%;%CD%\src`
3. Run the script: `python src\pipeline.py path\to\your_file.c --no-carbon --model results\pyg_gnn_v9\best_pyg_model.pt`

## Folder structure
- `src/` holds the backend code like the Flask app, pipeline, and models.
- `data/` and `dataset/` are where the graphs and datasets live.
- `results/` is where trained models and outputs are saved.
- `test_files/` has some sample code you can use for testing.
- `docs/` contains extra documentation.
