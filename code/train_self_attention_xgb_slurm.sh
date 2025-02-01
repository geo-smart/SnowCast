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
#SBATCH --ntasks-per-node=40                 # number of cores needed
#SBATCH --mem=50G
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
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

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

# ['date', 'lat', 'lon', 'SWE', 'station_name', 'swe_value', 'change_in_swe_inch', 'snow_depth', 'air_temperature_observed_f', 
#  'precipitation_amount', 'relative_humidity_rmin', 'potential_evapotranspiration', 'air_temperature_tmmx', 'relative_humidity_rmax', 
#  'mean_vapor_pressure_deficit', 'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect', 'Curvature', 'Northness', 'Eastness', 
#  'fsca', 'Slope', 'SWE_1', 'air_temperature_tmmn_1', 'potential_evapotranspiration_1', 'mean_vapor_pressure_deficit_1', 
#  'relative_humidity_rmax_1', 'relative_humidity_rmin_1', 'precipitation_amount_1', 'air_temperature_tmmx_1', 'wind_speed_1', 
#  'fsca_1', 'SWE_2', 'air_temperature_tmmn_2', 'potential_evapotranspiration_2', 'mean_vapor_pressure_deficit_2', 
#  'relative_humidity_rmax_2', 'relative_humidity_rmin_2', 'precipitation_amount_2', 'air_temperature_tmmx_2', 'wind_speed_2', 
#  'fsca_2', 'SWE_3', 'air_temperature_tmmn_3', 'potential_evapotranspiration_3', 'mean_vapor_pressure_deficit_3', 
#  'relative_humidity_rmax_3', 'relative_humidity_rmin_3', 'precipitation_amount_3', 'air_temperature_tmmx_3', 'wind_speed_3', 
#  'fsca_3', 'SWE_4', 'air_temperature_tmmn_4', 'potential_evapotranspiration_4', 'mean_vapor_pressure_deficit_4', 
#  'relative_humidity_rmax_4', 'relative_humidity_rmin_4', 'precipitation_amount_4', 'air_temperature_tmmx_4', 'wind_speed_4', 
#  'fsca_4', 'SWE_5', 'air_temperature_tmmn_5', 'potential_evapotranspiration_5', 'mean_vapor_pressure_deficit_5', 
#  'relative_humidity_rmax_5', 'relative_humidity_rmin_5', 'precipitation_amount_5', 'air_temperature_tmmx_5', 'wind_speed_5', 
#  'fsca_5', 'SWE_6', 'air_temperature_tmmn_6', 'potential_evapotranspiration_6', 'mean_vapor_pressure_deficit_6', 
#  'relative_humidity_rmax_6', 'relative_humidity_rmin_6', 'precipitation_amount_6', 'air_temperature_tmmx_6', 
#  'wind_speed_6', 'fsca_6', 'SWE_7', 'air_temperature_tmmn_7', 'potential_evapotranspiration_7', 'mean_vapor_pressure_deficit_7', 
#  'relative_humidity_rmax_7', 'relative_humidity_rmin_7', 'precipitation_amount_7', 'air_temperature_tmmx_7', 'wind_speed_7', 'fsca_7', 
#  'water_year', 'cumulative_SWE', 'cumulative_air_temperature_tmmn', 'cumulative_potential_evapotranspiration', 
#  'cumulative_mean_vapor_pressure_deficit', 'cumulative_relative_humidity_rmax', 'cumulative_relative_humidity_rmin', 
#  'cumulative_precipitation_amount', 'cumulative_air_temperature_tmmx', 'cumulative_wind_speed', 'cumulative_fsca']

def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print("df.columns: ", list(df.columns))
    df = df[df['swe_value'] > 0]
    df = df.sort_values(by=["station_name", "date"])
    
    # Select the first 10,000 rows
    # df = df.iloc[:100000]
    # print(f"Subset to first 10,000 rows. Data shape: {df.shape}")
    
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, features, target):
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.scaler = StandardScaler()

        # Normalize data
        self.data_normalized = self.normalize_data()
        
        # Create sequences
        self.sequences, self.targets = self.create_sequences()

    def normalize_data(self):
        # Normalize the features and target column
        print("Normalizing data...")
        features_data = self.data[self.features]
        target_data = self.data[self.target].values.reshape(-1, 1)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_data)
        
        # Normalize target
        target_normalized = self.scaler.fit_transform(target_data).flatten()

        # Create normalized dataframe
        normalized_data = self.data.copy()
        normalized_data[self.features] = features_normalized
        normalized_data[self.target] = target_normalized

        return normalized_data

    def create_sequences(self):
        sequences = []
        targets = []
        print(f"Creating sequences for {len(self.data['station_name'].unique())} unique stations.")
        for station in self.data['station_name'].unique():
            station_data = self.data[self.data['station_name'] == station]
            for i in range(len(station_data) - self.sequence_length):
                seq = station_data.iloc[i:i + self.sequence_length][self.features].values
                sequences.append(seq)
                targets.append(station_data.iloc[i + self.sequence_length][self.target])
        print(f"Total sequences created: {len(sequences)}")
        return np.array(sequences), np.array(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, hidden_dim))
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
#         print(f"Input shape: {x.shape}")
        # Add positional encoding
        x = self.input_embedding(x) + self.positional_encoding
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Pooling: Average across the sequence
        output = self.output_layer(x)
#         print(f"Output shape: {output.shape}")
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch
        
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                loss = criterion(output.squeeze(), y)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        
        # Calculate and print epoch time
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

    # Save the model with a timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join("/groups/ESS3/zsun/swe/", f"transformer_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return train_losses, val_losses

def evaluate_model(model, val_loader):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            y_true.extend(y.numpy())
            y_pred.extend(output.squeeze().numpy())
    
    # Compute RMSE and R2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    return y_true, y_pred

def plot_training_progress(train_losses, val_losses, plot_dir):
    plt.figure(figsize=(12, 12))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plot_dir, f'transformer_training_progress_{timestamp}.png'))

def plot_evaluation_results(y_true, y_pred, plot_dir):
    plt.figure(figsize=(12, 12))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted SWE')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plot_dir, f'transformer_predictions_vs_true_{timestamp}.png'))


def do_self_attention(filepath, plot_dir):
    # Load data
    df = load_data(filepath)

    # Prepare dataset and dataloaders
    print("Creating dataset and dataloaders...")
    print("input features: ", FEATURES)
    print("SEQUENCE_LENGTH: ", SEQUENCE_LENGTH)
    print("target column: ", TARGET_COLUMN)
    dataset = TimeSeriesDataset(df, SEQUENCE_LENGTH, FEATURES, TARGET_COLUMN)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32*20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32*20, shuffle=False)

    # Define model
    input_dim = len(FEATURES)
    hidden_dim = 64
    nhead = 4
    num_layers = 2
    output_dim = 1

    model = TransformerModel(input_dim, nhead, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 1000
    print(f"Starting training for {epochs} epochs...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

    # Plot training progress
    plot_training_progress(train_losses, val_losses, plot_dir)

    # Evaluate the model
    print("Evaluating the model...")
    y_true, y_pred = evaluate_model(model, val_loader)

    # Plot evaluation results
    plot_evaluation_results(y_true, y_pred, plot_dir)

# Main Script
if __name__ == "__main__":
    filepath = '/groups/ESS3/zsun/swe/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
    # Define paths and create necessary directories
    model_save_path = '/groups/ESS3/zsun/swe/transformer_model.pth'
    plot_dir = '/groups/ESS3/zsun/swe/plots'
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot directory ensured at: {plot_dir}")
    do_self_attention(filepath, plot_dir)


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


