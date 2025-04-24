import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA


# Load datasets
train = pd.read_csv("train.csv")

train['Period_Consumption_Index'] = train['Period_Consumption_Index'].round(2)


# Define the valid income levels
valid_income_levels = ['Low', 'Middle', 'Upper Middle', 'Rich']
# Replace invalid income levels with 'Unknown'
train['Income_Level'] = train['Income_Level'].apply(lambda x: x if x in valid_income_levels else 'Unknown')

# Check for NaN values in the 'Guests' column
nan_count = train['Appliance_Usage'].isna().sum()

# Print the number of NaN values in the 'Guests' column
print(f'Number of NaN values in the "Appliance_Usage" column: {nan_count}')

# Check for negative values in the 'Guests' column
negative_values_count = (train['Guests'] < 0).sum()

# Print the number of negative values in the 'Guests' column
print(f'Number of negative values in the "Guests" column: {negative_values_count}')


# Replace negative values in the 'Guests' column with 0
train['Guests'] = train['Guests'].apply(lambda x: 0 if x < 0 else x)

# Replace NaN values in 'Appliance_Usage' with the mode (most frequent value)
mode_value = train['Appliance_Usage'].mode()[0]  # Get the most frequent value
train['Appliance_Usage'] = train['Appliance_Usage'].fillna(mode_value)



# Strip any extra spaces in 'Apartment_Type' column
train['Apartment_Type'] = train['Apartment_Type'].str.strip()
# train['Residents'] = train['Residents'].str.strip()

# Define a function to fill in the apartment type based on the number of residents
def assign_apartment_type(row):
    if pd.isnull(row['Apartment_Type']) or row['Apartment_Type'] == '': # Only update if the Apartment_Type is NaN
        if row['Residents'] == 1:
            return 'Studio'
        elif row['Residents'] in [2,3]:
            return '1BHK'
        elif row['Residents'] in [4, 5]:
            return '2BHK'
    return row['Apartment_Type']  # Keep the original value if it is not NaN

# Apply the function to the 'Apartment_Type' column
train['Apartment_Type'] = train.apply(assign_apartment_type, axis=1)


# Check if there are still any empty 'Apartment_Type' rows
empty_apartments = train[train['Apartment_Type'].isnull()]
if not empty_apartments.empty:
    print(f"There are still {empty_apartments.shape[0]} rows with empty 'Apartment_Type'.")



# # Define a function to update 'Residents' based on 'Apartment_Type' for negative values
def update_negative_residents(row):
    if row['Residents'] < 0:  # Only update if 'Residents' is negative
        if row['Apartment_Type'] == 'Detached':
            return 4
        elif row['Apartment_Type'] == 'Studio':
            return 1
        elif row['Apartment_Type'] == '1BHK':
            return 2
        elif row['Apartment_Type'] == '2BHK':
            return 3
        elif row['Apartment_Type'] == '3BHK':
            return 4
        elif row['Apartment_Type'] == 'Cottage':
            return 3
        elif row['Apartment_Type'] == 'Bungalow':
            return 3
    return row['Residents']  # Keep the original value if not negative
# Apply the function to the 'Residents' column
train['Residents'] = train.apply(update_negative_residents, axis=1)


# Remove rows where 'Humidity' is NaN or non-numeric
train = train[train['Humidity'].apply(pd.to_numeric, errors='coerce').notna()] 

# Assuming 'train' is your dataset and 'Timestamp' is the column
train['Timestamp_1'] = pd.to_datetime(train['Timestamp'], format='%d/%m/%Y %H')
# Check the conversion
print(train['Timestamp_1'].head())
# Extracting various time-based features
train['Hour'] = train['Timestamp_1'].dt.hour
train['Day'] = train['Timestamp_1'].dt.day
train['Month'] = train['Timestamp_1'].dt.month
train['Weekday'] = train['Timestamp_1'].dt.weekday  # 0 = Monday, 6 = Sunday
train['Is_Weekend'] = train['Weekday'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if weekend, 0 if weekday

# If you have data spanning multiple years, extract the year as well
train['Year'] = train['Timestamp_1'].dt.year

# Drop Timestamp column as we no longer need it for modeling
train = train.drop(columns=['Timestamp_1'])
def classify_day_night(hour):
    return 1 if 8 <= hour < 20 else 0
#Classify Summer
def classify_summer(month):
    return 1 if month in [6, 7, 8] else 0

train["Is_Summer"] = train["Month"].apply(classify_summer)
# Apply the classification
train["Is_Day"] = train["Hour"].apply(classify_day_night)


# Check the first few rows to confirm feature extraction
print(train.head())
train['Humidity'] = pd.to_numeric(train['Humidity'], errors='coerce')


# Calculate the mean of the 'water price' column (ignoring negative values)
mean_water_price = train[train['Water_Price'] >= 0]['Water_Price'].mean()

# Replace negative values in the 'water price' column with the mean value
train['Water_Price'] = train['Water_Price'].apply(lambda x: mean_water_price if x < 0 else x)

"""---------------------------Model for Income Level-----------------------------------"""
# Encode categorical features (Apartment_Type)
label_enc = LabelEncoder()
train['Apartment_Type_Encode'] = label_enc.fit_transform(train['Apartment_Type'])

# Separate known and unknown income levels
known_data = train[train['Income_Level'] != 'Unknown']
unknown_data = train[train['Income_Level'] == 'Unknown']

# Encode Income Level for training
income_encoder = LabelEncoder()
known_data['Income_Level_Encoded'] = income_encoder.fit_transform(known_data['Income_Level'])

# Define features and target
X = known_data[['Apartment_Type_Encode']]
y = known_data['Income_Level_Encoded']

# Train a model
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=42)
model_i = RandomForestClassifier()
model_i.fit(X_train_i, y_train_i)

# Predict on test set
y_pred = model_i.predict(X_test_i)

# Calculate Accuracy
accuracy = accuracy_score(y_test_i, y_pred)

# Print Accuracy
print(f"Accuracy: {accuracy:.4f}")

# Predict for unknown values
X_unknown = unknown_data[['Apartment_Type_Encode']]
predicted_income_encoded = model_i.predict(X_unknown)

# Convert predictions back to labels
unknown_data['Income_Level'] = income_encoder.inverse_transform(predicted_income_encoded)

# Merge back into the original dataset
train.loc[train['Income_Level'] == 'Unknown', 'Income_Level'] = unknown_data['Income_Level']

"""---------------------------Model for Income Level-----------------------------------"""

"""-----------------------------------LabelEncoding----------------------------------------"""
# Define the custom order
categories_i = [['Low', 'Middle', 'Upper Middle', 'Rich']]
categories_a = [['Unknown','Garden','Fountain','Jacuzzi','Swimming Pool']]


# Apply Ordinal Encoding
encoder_i = OrdinalEncoder(categories=categories_i)
train['Income_Level_Encoded'] = encoder_i.fit_transform(train[['Income_Level']])
# Apply Ordinal Encoding
train['Amenities'] = train['Amenities'].fillna('Unknown')
encoder_a = OrdinalEncoder(categories=categories_a)
train['Amenities_Encoded'] = encoder_a.fit_transform(train[['Amenities']])
"""-----------------------------------LabelEncoding----------------------------------------"""


"""-----------------------------------Model_1 for Temperature-------------------------------------------"""
# Check the number of missing values in 'Temperature'
print(f"Missing Temperature values before imputation: {train['Temperature'].isna().sum()}")

# Separate rows with missing and non-missing Temperature
train_non_missing = train.dropna(subset=['Temperature'])  # Rows with no missing temperature
train_missing = train[train['Temperature'].isna()]  # Rows with missing temperature

# Features for prediction (in this case, 'Humidity' and other relevant columns)
# You can add more features depending on your dataset
features = ['Humidity', 'Is_Day', 'Day','Year','Period_Consumption_Index','Water_Price','Water_Consumption']

# Train a linear regression model on the rows with non-missing temperature
X_train = train_non_missing[features]  # Features from rows with known temperature
y_train = train_non_missing['Temperature']  # Known temperature values

# Initialize the model
model = RandomForestRegressor(n_estimators=300, random_state=42)
# model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Use the model to predict the missing temperature values
X_missing = train_missing[features]  # Features from rows with missing temperature
predicted_temp = model.predict(X_missing)

# Fill the missing temperature values with the predicted values
train.loc[train['Temperature'].isna(), 'Temperature'] = predicted_temp

# Verify if the missing values have been filled
print(f"Missing Temperature values after imputation: {train['Temperature'].isna().sum()}")
train['Temperature'] = train['Temperature'].round(2)


# Evaluate the model on the non-missing data
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
print(f"Mean Squared Error Temperature: {mse}")


"""-----------------------------------Model_1 for Temperature-------------------------------------------"""
# Check the shape of the cleaned dataframe to verify the removal
print(f"Shape of the cleaned DataFrame: {train.shape}")
train.to_csv("updated_train.csv", index=False)



