# Water Monitoring System

## 📁 Folder Contents

- `water_env/` – Python environment for the project  
- `Data_Preprocessing.py` – Data cleaning and feature engineering using `train.csv`  
- `ml.py` – Final model for water consumption prediction  
- `requirements.txt` – Required Python libraries  
- `submission.csv` – Final predicted water consumption values  
- `updated_train.csv` – Cleaned dataset used for modeling  

---

## 🧠 Project Description

The goal of this project is to develop a **machine learning model** that predicts **daily water consumption** for individual households. The model leverages:
- Historical water usage patterns  
- Household characteristics  
- Weather conditions  
- Conservation behaviors  

---

## 🔧 Data Preprocessing Pipeline (`Data_Preprocessing.py`)

### 1. Handling Missing or Invalid Values
- **Rounded `Period_Consumption_Index`** to two decimal places.
- Replaced invalid `Income_Level` entries with `"Unknown"`.
- Filled missing `Appliance_Usage` with the column's **mode**.
- Replaced negative `Guests` values with **0**.
- Removed non-numeric values in the `Humidity` column.

### 2. Feature Transformation
- Trimmed whitespace in `Apartment_Type`.
- Imputed missing `Apartment_Type` based on `Residents`.
- Corrected invalid `Residents` values using logic tied to `Apartment_Type`.

### 3. Date & Time Processing
- Converted `Timestamp` to `datetime`, then derived:
  - `Hour`, `Day`, `Month`, `Weekday`, `Year`
  - Boolean flags: `Is_Weekend`, `Is_Summer`, `Is_Day`

### 4. Imputation via ML
- Used `RandomForestRegressor` to impute missing `Temperature` values using related features.

### 5. Feature Encoding
- **Label Encoding** for `Apartment_Type` and `Income_Level`.
- **Ordinal Encoding** for ordered categories like `Amenities`.

### 6. Predictive Imputation
- Predicted unknown `Income_Level` using `RandomForestClassifier` based on `Apartment_Type`.
- Trained model to predict missing `Temperature` using available weather features.

### 7. Final Cleanup
- Dropped intermediate columns like `Timestamp_1`.

### 8. Export Cleaned Data
- Cleaned dataset saved to `updated_train.csv`.

---

## 📊 Water Consumption Prediction (`ml.py`)

### ⚙️ Model: XGBoost Regressor

### Steps Involved:

#### 1. Dataset Preparation
- Load cleaned dataset from `updated_train.csv`.

#### 2. Feature/Target Definition
- **Target:** `Water_Consumption`  
- **Features:** All columns except `Water_Consumption`, `Timestamp`, `Income_Level`, `Apartment_Type`, `Amenities`.

#### 3. Train-Test Split
- 80/20 split using `train_test_split()`.

#### 4. Hyperparameter Tuning
- Grid search (`GridSearchCV`) over:
  - `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`

#### 5. Training
- Model trained using the best parameters found by `GridSearchCV`.

#### 6. Evaluation
- Metrics:
  - **MAE** – Mean Absolute Error
  - **RMSE** – Root Mean Squared Error

#### 7. Future Predictions
- Forecasts water consumption for the next **6000 entries** using most recent data.

#### 8. Output
- Saves predictions with timestamps to `submission.csv`.

---

## 📌 Key Highlights

- Uses **XGBoost Regressor** for efficient and accurate prediction.
- Incorporates **machine learning imputation** for missing values.
- Employs **GridSearchCV** for hyperparameter optimization.
- Supports **future forecasting** based on historical trends.

---

## 🛠️ Installation Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
