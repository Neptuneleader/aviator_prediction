import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Collect historical data of Aviator game outcomes
# This can be done by scraping the casino's website or by accessing their API
# For example, let's assume we have a CSV file called 'aviator_data.csv' with the following columns:
# 'PlayerID', 'BetAmount', 'WinAmount', 'Result'
data = pd.read_csv('aviator_data.csv')

# Step 2: Preprocess the data
# Convert categorical variables to numerical
data['Result'] = data['Result'].map({'Win': 1, 'Loss': 0})

# Split the data into features (X) and target (y)
X = data[['PlayerID', 'BetAmount']]
y = data['Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 5: Make predictions for new data
# Assuming we have a new player with ID 1234 and bet amount of 100
new_player = pd.DataFrame({'PlayerID': [1234], 'BetAmount': [100]})
prediction = model.predict(new_player)
print(f"Predicted probability of winning: {prediction[0]:.2f}")

