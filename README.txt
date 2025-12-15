# Loan Default Prediction API

## Environment Assumptions

## It is assumed that the reviewer has **Python (>=3.9)** and either **pip or conda** installed locally.

## If Python is not installed:
## - Python can be downloaded from https://www.python.org/downloads/
## - Conda (recommended) can be installed from https://docs.conda.io/en/latest/miniconda.html

## All steps below are intended to be run using your **local command line interface (Terminal on ## macOS/Linux, Command Prompt or PowerShell on Windows)**.

### 1. Create environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

### 2. Install dependencies
pip install -r requirements.txt

# If xgboost import fails please use conda forge
conda install -c conda-forge xgboost

### 3. Run the API
python -m uvicorn app:app --reload

### 4. Open API docs
http://127.0.0.1:8000/docs
