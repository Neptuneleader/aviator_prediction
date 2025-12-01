import argparse
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


REQUIRED_COLUMNS = {"PlayerID", "BetAmount", "WinAmount", "Result"}


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read CSV at '{path}': {e}")
        sys.exit(1)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"CSV missing required columns: {sorted(missing)}")
        sys.exit(1)

    if not set(df["Result"].unique()).issubset({"Win", "Loss"}):
        print("'Result' must contain only 'Win' or 'Loss'")
        sys.exit(1)

    df["Result"] = df["Result"].map({"Win": 1, "Loss": 0})
    return df


def train_and_evaluate(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    player_id: int = 1234,
    bet_amount: float = 100,
    scale_features: bool = True,
    metrics_output: str = "metrics.json",
    report_output: str | None = "report.txt",
    features: list[str] | None = None,
    save_model: str | None = None,
):
    df = load_data(data_path)

    if not features:
        features = ["PlayerID", "BetAmount"]
    for col in features:
        if col not in df.columns:
            print(f"Feature '{col}' not found in CSV.")
            sys.exit(1)

    X = df[features]
    y = df["Result"]

    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float('nan')
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    new_player = pd.DataFrame({"PlayerID": [player_id], "BetAmount": [bet_amount]})
    win_proba = model.predict_proba(new_player)[:, 1]
    print(f"Predicted probability of winning: {win_proba[0]:.2f}")

    metrics = {"accuracy": float(acc), "roc_auc": float(auc)}
    if metrics_output:
        try:
            with open(metrics_output, "w", encoding="utf-8") as f:
                json.dump(metrics, f)
            print(f"Saved metrics to {metrics_output}")
        except Exception as e:
            print(f"Failed to save metrics to {metrics_output}: {e}")

    if report_output:
        try:
            with open(report_output, "w", encoding="utf-8") as f:
                f.write("Confusion Matrix\n")
                f.write(str(cm) + "\n\n")
                f.write("Classification Report\n")
                f.write(classification_report(y_test, y_pred, digits=3))
            print(f"Saved report to {report_output}")
        except Exception as e:
            print(f"Failed to save report to {report_output}: {e}")

    if save_model:
        try:
            dump({
                "model": model,
                "features": features,
                "scaler": scaler if scale_features else None,
            }, save_model)
            print(f"Saved model artifact to {save_model}")
        except Exception as e:
            print(f"Failed to save model to {save_model}: {e}")

    return model, features, scaler if scale_features else None


def predict_from_artifact(artifact_path: str, player_id: int, bet_amount: float) -> float:
    try:
        bundle = load(artifact_path)
    except Exception as e:
        print(f"Failed to load model artifact from '{artifact_path}': {e}")
        sys.exit(1)

    model = bundle.get("model")
    features = bundle.get("features")
    scaler = bundle.get("scaler")

    if not model or not features:
        print("Artifact missing model or features.")
        sys.exit(1)

    row = {"PlayerID": player_id, "BetAmount": bet_amount}
    X_new = pd.DataFrame([[row.get(col, 0) for col in features]], columns=features)
    if scaler is not None:
        X_new = scaler.transform(X_new)
    proba = model.predict_proba(X_new)[:, 1][0]
    return float(proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aviator prediction classifier baseline")
    parser.add_argument("--data", required=True, help="Path to aviator_data.csv")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified split")
    parser.add_argument("--player-id", type=int, default=1234, help="Sample prediction PlayerID")
    parser.add_argument(
        "--bet-amount", type=float, default=100, help="Sample prediction BetAmount"
    )
    parser.add_argument("--no-scale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--output", default="metrics.json", help="Path to save metrics JSON (blank to disable)")
    parser.add_argument("--report", default="report.txt", help="Path to save text report (blank to disable)")
    parser.add_argument("--features", nargs="*", default=None, help="Feature columns to use (default: PlayerID BetAmount)")
    parser.add_argument("--save-model", default=None, help="Path to save trained model artifact (joblib)")
    parser.add_argument("--predict-artifact", default=None, help="Path to a saved model to predict with")
    args = parser.parse_args()
    if args.predict_artifact:
        proba = predict_from_artifact(args.predict_artifact, args.player_id, args.bet_amount)
        print(f"Predicted probability of winning: {proba:.2f}")
        sys.exit(0)

    train_and_evaluate(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify,
        player_id=args.player_id,
        bet_amount=args.bet_amount,
        scale_features=not args.no_scale,
        metrics_output=(args.output or None),
        report_output=(args.report or None),
        features=args.features,
        save_model=args.save_model,
    )
