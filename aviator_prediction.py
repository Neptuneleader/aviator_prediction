import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Result" not in df.columns:
        raise ValueError("CSV must include a 'Result' column with Win/Loss")
    df["Result"] = df["Result"].map({"Win": 1, "Loss": 0})
    return df


def main(data_path: str):
    df = load_data(data_path)

    X = df[["PlayerID", "BetAmount"]]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    new_player = pd.DataFrame({"PlayerID": [1234], "BetAmount": [100]})
    win_proba = model.predict_proba(new_player)[:, 1]
    print(f"Predicted probability of winning: {win_proba[0]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aviator prediction baseline")
    parser.add_argument(
        "--data", required=True, help="Path to aviator_data.csv"
    )
    args = parser.parse_args()
    main(args.data)
