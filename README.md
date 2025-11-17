# CS 634 — Final Term Project 

## Setup (one time)

```bash
# Create & activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project layout
- data/ — CSV dataset - WISDM_raw.csv
- notebook/ — `finaltermproject.ipynb` 
- report/ — `thakre_srushti_finaltermproject/pdf` 
- src/ — source code  
  - `build_wisdm_raw_csv.py` — prepares trimmed CSV from raw dataset
  - `run_wisdm_cv.py` — runs 10-fold GroupKFold CV for RF, SVM, and GRU

## Report
See **[CS634 Final Term Project Report](report/thakre_srushti_finaltermproject.pdf)**.
