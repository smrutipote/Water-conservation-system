import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
df = pd.read_csv("updated_train.csv")  # Update with your actual file path
# Check if Timestamp column exists
timestamp_exists = "Timestamp" in df.columns
# Define features and target variable
X = df.drop(columns=["Water_Consumption",'Income_Level', 'Apartment_Type', 'Amenities'])  # Drop non-relevant columns
y = df["Water_Consumption"]


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=["Timestamp"]), y, test_size=0.2, random_state=42, shuffle=False)


# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.09],
    'max_depth': [3, 6, 10],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Train XGBoost model
# model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, objective='reg:squarederror')
# model.fit(X_train, y_train)

grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=4, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Make predictions
predictions = best_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"MAE: {mae}, RMSE: {rmse}")

import pandas as pd
import matplotlib.pyplot as plt

# Predict next 6000 values (assuming sequential future timestamps)
# Adjust based on feature engineering and data you have.
# For example, X might be your feature set, and `grid_search` is your trained model
future_X = X.tail(6000).copy()  # Adjust based on your data
y_future_pred = grid_search.predict(future_X.drop(columns=["Timestamp"]))  # Exclude Timestamp during prediction

# # Load the test.csv file to get the timestamps and features
test_df = pd.read_csv('test.csv')

# Extract the relevant timestamps from the test data
# Assuming the Timestamp column is already in the format you need
timestamps_t = test_df["Timestamp"]


# Save predictions with timestamp
future_predictions = pd.DataFrame({
    "Timestamp": timestamps_t, 
    "Water_Consumption": y_future_pred
})

# Save the DataFrame to a CSV file
future_predictions.to_csv("submission.csv", index=False)
print(f"Shape of the cleaned DataFrame: {future_predictions.shape}")

