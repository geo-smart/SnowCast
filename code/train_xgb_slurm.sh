#!/bin/bash
# Specify the name of the script you want to submit
SCRIPT_NAME="train_self_attention_xgb_slurm.sh"
echo "write the slurm script into ${SCRIPT_NAME}"

# CPU
#SBATCH --qos=qtong             #
#SBATCH --partition=contrib     # partition (queue): debug, interactive, contrib, normal, orc-test

# GPU
#SBATCH --account=qtong
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:1                # up to 8; only request what you need


cat > ${SCRIPT_NAME} << EOF
#!/bin/bash
#SBATCH -J self_attention_xgb_slurm       # Job name
#SBATCH --qos=qtong             #
#SBATCH --partition=contrib     # partition (queue): debug, interactive, contrib, normal, orc-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20                 # number of cores needed
#SBATCH --mem=150G
#SBATCH --time=24:00:00         # walltime
#SBATCH --mail-user=zsun@gmu.edu    #Email account
#SBATCH --mail-type=FAIL           #When to email
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`

set echo
umask 0027


# Activate your customized virtual environment
source /home/zsun/anaconda3/bin/activate

export CUDA_LAUNCH_BLOCKING=1

python -u << INNER_EOF
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
    'SWE', 'change_in_swe_inch', 'snow_depth', 'air_temperature_observed_f', 
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
def train_xgboost(X_train, y_train, X_val, y_val):
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=True)

    # Save the model with a timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join("/groups/ESS3/zsun/swe/", f"xgb_model_{timestamp}.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

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
    df = df[df[target_col] > 0]  # Keep positive SWE values
    # df = df.sample(n=500000, random_state=42)  # Downsample for speed
    print(f"Filtered dataset shape: {df.shape}")

    # Drop all columns with "cumulative" in their names
    cumulative_columns = [col for col in df.columns if 'cumulative' in col.lower()]
    X = df.drop(columns=[target_col, 'date', 'station_name'] + cumulative_columns)

    print(f"Dropped columns: {cumulative_columns}")

    # X = df.drop(columns=[target_col, 'date', 'station_name'])  # Drop unnecessary columns
    y = df[target_col]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).squeeze()

    return X_scaled, y_scaled, scaler_y


def do_xgboost(filepath, plot_dir):
    # Load and preprocess data
    print(f"Loading and preprocessing data from: {filepath}")
    X, y, scaler_y = load_preprocess_data_xgboost(filepath)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # Evaluate model
    print("Evaluating XGBoost model...")
    evaluate_xgboost(xgb_model, X_test, y_test, scaler_y, plot_dir)

# Main Script
if __name__ == "__main__":
    filepath = '/groups/ESS3/zsun/swe/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
    # Define paths and create necessary directories
    model_save_path = '/groups/ESS3/zsun/swe/transformer_model.pth'
    plot_dir = '/groups/ESS3/zsun/swe/plots'
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot directory ensured at: {plot_dir}")
    # do_self_attention(filepath, plot_dir)
    do_xgboost(filepath, plot_dir)


INNER_EOF

EOF

# Submit the Slurm job and wait for it to finish
echo "sbatch ${SCRIPT_NAME}"


# Submit the Slurm job
job_id=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')
echo "job_id="${job_id}

if [ -z "${job_id}" ]; then
    echo "job id is empty. something wrong with the slurm job submission."
    exit 1
fi

# Wait for the Slurm job to finish
file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
previous_content=$(<"${file_name}")
exit_code=0
while true; do
    # Capture the current content
    file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
    current_content=$(<"${file_name}")

    # Compare current content with previous content
    diff_result=$(diff <(echo "$previous_content") <(echo "$current_content"))
    # Check if there is new content
    if [ -n "$diff_result" ]; then
        echo "$diff_result"
    fi
    # Update previous content
    previous_content="$current_content"

    job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
    if [[ $job_status == *"COMPLETED"* || $job_status == *"CANCELLED"* || $job_status == *"FAILED"* || $job_status == *"TIMEOUT"* || $job_status == *"NODE_FAIL"* || $job_status == *"PREEMPTED"* || $job_status == *"OUT_OF_MEMORY"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        break;
    fi
    sleep 100  # Adjust the sleep interval as needed
done

echo "Slurm job ($job_id) has finished."

echo "Print the job's output logs"
sacct --format=JobID,JobName,State,ExitCode,MaxRSS,Start,End -j $job_id
#find /scratch/zsun/ -type f -name "*${job_id}.out" -exec cat {} \;

echo "All slurm job for ${SCRIPT_NAME} finishes."

job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
echo "job status $job_status"
if [[ $job_status == *"COMPLETED"* ]]; then
    exit 0
fi

exit 1


