## Aviator Prediction

This repository contains a minimal example for training and evaluating a simple model using historical Aviator outcomes from a CSV file.

![CI](https://github.com/lunifermoon89-ux/aviator_prediction/actions/workflows/ci.yml/badge.svg)

### Data
Expected CSV: `aviator_data.csv` with columns:
- `PlayerID`, `BetAmount`, `WinAmount`, `Result`

`Result` should be categorical: `Win` or `Loss`.

### Quick Start
1. Create or place `aviator_data.csv` in the repo root (see sample below).
2. Install requirements:
	```bash
	pip install -r requirements.txt
	```
3. Run the script:
	```bash
	python aviator_prediction.py --data aviator_data.csv
	```
4. Optional flags:
	```bash
	python aviator_prediction.py --data aviator_data.csv --test-size 0.3 --seed 123 --player-id 1001 --bet-amount 50 --output metrics.json --report report.txt
	```

### Model & Metrics
- Converts `Result` to 1/0 and trains a `LogisticRegression` classifier.
- Prints `Accuracy` and `ROC-AUC` on a test split, plus Confusion Matrix and Classification Report.
- Outputs predicted win probability for a sample player (`PlayerID=1234`, `BetAmount=100`).
- Saves metrics to `metrics.json` for downstream use.

### Sample CSV
`sample_aviator_data.csv` example:
```csv
PlayerID,BetAmount,WinAmount,Result
1001,50,0,Loss
1002,20,40,Win
1003,100,0,Loss
1004,10,18,Win
```

Use your own `aviator_data.csv` for real runs.

### Development
- Tests: `pytest`
- CI: GitHub Actions runs the script and tests on every PR/push.
 - Feature scaling is enabled by default; disable with `--no-scale`.


