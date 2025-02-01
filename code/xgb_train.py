import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from snowcast_utils import homedir, work_dir, data_dir, plot_dir, model_dir

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define sequence length and target column
SEQUENCE_LENGTH = 7
TARGET_COLUMN = 'swe_value'

# Base features (without lags)
BASE_FEATURES = [
#     'SWE', 'air_temperature_tmmn', 'relative_humidity_rmin',
#     'precipitation_amount', 'wind_speed', 'fsca'
    'SWE', 
    # 'change_in_swe_inch', 'snow_depth', 'air_temperature_observed_f',  # these are from SNOTEL stations
    'precipitation_amount', 'relative_humidity_rmin', 'potential_evapotranspiration', 'air_temperature_tmmx', 'relative_humidity_rmax', 
    'mean_vapor_pressure_deficit', 'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect', 'Curvature', 'Northness', 'Eastness', 
    'fsca', 'Slope', 
]

# Define paths and create necessary directories
# filepath = '/groups/ESS3/zsun/swe/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'

# Features to include in the sequence (including lagged features)
# FEATURES = BASE_FEATURES + [f"{feature}_{i}" for feature in BASE_FEATURES for i in range(1, SEQUENCE_LENGTH + 1)]
FEATURES = BASE_FEATURES

# Train XGBoost model
def train_xgboost(X_train, y_train, X_val, y_val, model_save_path):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Save the model with a timestamped filename
    model_dir = os.path.dirname(model_save_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_bkp_path = os.path.join(model_dir, f"xgb_model_{timestamp}.json")
    model.save_model(model_bkp_path)
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path} and backed up to {model_bkp_path}")

    return model

# Evaluate XGBoost model
def evaluate_xgboost(model, X_test, y_test, scaler_y, plot_dir):
    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    true_values = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()

    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    print(f"XGBoost Model Evaluation:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot Predictions vs True Values
    plt.figure(figsize=(10, 10))
    plt.scatter(true_values, predictions, alpha=0.6)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('XGBoost Predictions vs True Values')
    plt.grid(True)
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("XGBoost model is saved to ", os.path.join(plot_dir, f'xgboost_predictions_vs_true_{timestamp}.png'))
    plt.savefig(os.path.join(plot_dir, 'xgboost_predictions_vs_true.png'))
    plt.close()

    # Residuals plot
    residuals = true_values - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='green', alpha=0.7)
    plt.xlabel('Residuals')
    plt.title('XGBoost Residuals Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'xgboost_residuals_{timestamp}.png'))
    plt.close()


# Load and preprocess data
def load_preprocess_data_xgboost(filepath, target_col='swe_value'):
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")
    print("df.columns = ", list(df.columns))
    df = df.fillna(-1)
    df = df[df[target_col] >= 0]  # Keep positive SWE values

    print("Assign -1 to fsca and SWE column where values are >100..")
    df.loc[df['fsca'] > 100, 'fsca'] = -1
    df.loc[df['SWE'] > 100, 'SWE'] = -1
    # df = df.sample(n=500000, random_state=42)  # Downsample for speed
    print(f"Filtered dataset shape: {df.shape}")

    # Drop all columns with "cumulative" in their names
    cumulative_columns = [col for col in df.columns if 'cumulative' in col.lower()]

    precip_columns = [col for col in df.columns if 'precipitation_amount' in col.lower()]

    columns_to_drop = [
        'station_name', 
        # 'swe_value', 
        'change_in_swe_inch', 
        'snow_depth', 
        'air_temperature_observed_f', 
        # 'precipitation_amount'
    ] + cumulative_columns + precip_columns

    X = df.drop(columns=[target_col, 'date', ] + columns_to_drop)
    print(f"Dropped columns: {columns_to_drop}")
    print("Only choose the desired features, remove the duplicated time series columns")
    # print(FEATURES)
    # X = X[FEATURES]
    print("Current X columns: ", X.columns)
    
    # X = df.drop(columns=[target_col, 'date', 'station_name'])  # Drop unnecessary columns
    y = df[target_col]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).squeeze()

    return X_scaled, y_scaled, scaler_y


def do_xgboost(filepath, plot_dir, model_save_path):
    # Load and preprocess data
    print(f"Loading and preprocessing data from: {filepath}")
    X, y, scaler_y = load_preprocess_data_xgboost(filepath)
    print(f"Initial data shapes - X: {X.shape}, y: {y.shape}")
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    print(f"After first split - X_train: {X_train.shape}, y_train: {y_train.shape}, X_temp: {X_temp.shape}, y_temp: {y_temp.shape}")
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"After second split - X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, model_save_path)

    # Evaluate model
    print("Evaluating XGBoost model...")
    evaluate_xgboost(xgb_model, X_test, y_test, scaler_y, plot_dir)

# Main Script
if __name__ == "__main__":
    filepath = f'{work_dir}/all_points_final_merged_training.csv'
    # Define paths and create necessary directories
    model_save_path = f'{model_dir}/xgb_alone_model.pth'
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot directory ensured at: {plot_dir}")
    # do_self_attention(filepath, plot_dir)
    do_xgboost(filepath, plot_dir, model_save_path)


