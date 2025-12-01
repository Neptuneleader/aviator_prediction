import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    new_player = pd.DataFrame({"PlayerID": [1234], "BetAmount": [100]})
    prediction = model.predict(new_player)
    print(f"Predicted probability of winning: {prediction[0]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aviator prediction baseline")
    parser.add_argument(
        "--data", required=True, help="Path to aviator_data.csv"
    )
    args = parser.parse_args()
    main(args.data)
