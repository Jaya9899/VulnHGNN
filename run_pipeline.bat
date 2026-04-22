@echo off
cd /d "c:\Users\jlux1\OneDrive\Documents\VulnerabilityDetectionGNN"
set PYTHONPATH=%PYTHONPATH%;%CD%\src
python src\pipeline.py test_files\test_02_cwe190_overflow.c --no-carbon --model results\pyg_gnn_v9\best_pyg_model.pt
