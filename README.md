## Aviator Prediction

This repository contains a minimal example for training and evaluating a simple model using historical Aviator outcomes from a CSV file.

### Data
Expected CSV: `aviator_data.csv` with columns:
- `PlayerID`, `BetAmount`, `WinAmount`, `Result`

`Result` should be categorical: `Win` or `Loss`.

### Quick Start
1. Create or place `aviator_data.csv` in the repo root.
2. Install requirements:
	```bash
	pip install -r requirements.txt
	```
3. Run the script:
	```bash
	python aviator_prediction.py --data aviator_data.csv
	```

### Notes
- The example converts `Result` to 1/0 and trains a simple linear regression baseline.
- Replace or extend the model as needed; consider additional features (e.g., session, timing, odds) for meaningful predictions.


