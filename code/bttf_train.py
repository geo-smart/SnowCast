
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
from snowcast_utils import homedir, work_dir

matplotlib.use('Agg')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
SEQUENCE_LENGTH = 7
TARGET_COLUMN = 'swe_value'

BASE_FEATURES = [
    'SWE', 
    #'change_in_swe_inch', 'snow_depth', 'air_temperature_observed_f',
    'precipitation_amount', 'relative_humidity_rmin', 'potential_evapotranspiration',
    'air_temperature_tmmx', 'relative_humidity_rmax', 'mean_vapor_pressure_deficit',
    'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect', 'Curvature',
    'Northness', 'Eastness', 'fsca', 'Slope'
]

# ET can reach 0.82 and 2.2
# selected_columns = [
#   'swe_value',
#   'SWE',
#   'fsca',
#   'air_temperature_tmmx', 
#   'air_temperature_tmmn', 
#   'potential_evapotranspiration', 
#   'relative_humidity_rmax', 
#   'Elevation',	
#   'Slope',	
#   'Curvature',	
#   'Aspect',	
#   'Eastness',	
#   'Northness',
# ]

# Derived features include lagged values
# FEATURES = BASE_FEATURES + [f"{feature}_lag{i}" for feature in BASE_FEATURES for i in range(1, SEQUENCE_LENGTH + 1)]
FEATURES = BASE_FEATURES

# Data Preprocessing
def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    df = df[df[TARGET_COLUMN] > 0]
    df = df.sort_values(by=["station_name", "date"])

    print(df.columns)
    df = df.fillna(-1)
    print("Assign -1 to fsca and SWE column where values are >100..")
    df.loc[df['fsca'] > 100, 'fsca'] = -1
    df.loc[df['SWE'] > 100, 'SWE'] = -1
    
    # df = df.iloc[:10000]  # Limit rows for faster processing
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, features, target_feature, forecast_horizon=7):
        """
        Initializes the TimeSeriesDataset object.

        Args:
            data (DataFrame): The data used for training.
            sequence_length (int): The length of the historical time window (e.g., past 7 days).
            features (list): List of feature columns to use as input.
            target_feature (str): The specific target feature to predict (e.g., 'SWE').
            forecast_horizon (int): The number of days to forecast for each day in the input sequence.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.target_feature = target_feature
        self.forecast_horizon = forecast_horizon
        self.sequences, self.targets = self.create_sequences()

    def create_sequences(self):
        sequences = []
        targets = []

        # Iterate over each station (or other unique identifier in your dataset)
        for station in self.data['station_name'].unique():
            station_data = self.data[self.data['station_name'] == station]

            # Ensure sufficient data for both input and output sequences
            for i in range(len(station_data) - self.sequence_length - self.forecast_horizon):
                # Input sequence: past `sequence_length` days
                seq = station_data.iloc[i:i + self.sequence_length][self.features].values
                sequences.append(seq)

                # Target sequence: for each day in the input sequence, predict the next `forecast_horizon` days
                targets.append(
                    station_data.iloc[i + self.sequence_length:i + self.sequence_length + \
                                      self.forecast_horizon][self.target_feature].values
                )

        # Return sequences and targets as numpy arrays
        return np.array(sequences), np.array(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Fetch the input sequence and target sequence
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # Print the shapes for debugging
#         print(f"Input sequence shape: {sequence.shape}, Target sequence shape: {target.shape}")
        
        return sequence, target


# Transformer Feature Extractor
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, num_layers, output_dim):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.feature_pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence length
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Add positional encoding to the input
        x = self.input_embedding(x) + self.positional_encoding
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        # Pool across the sequence dimension
        x = self.feature_pool(x.transpose(1, 2)).squeeze(-1)
        return self.output_layer(x)



# Training and Feature Extraction
def train_transformer_feature_extractor(model, train_loader, val_loader, criterion, optimizer, epochs):
    """
    Trains the transformer model with the given data and returns the training
    and validation loss over epochs.
    """
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)  # Calculate average training loss
        train_loss_list.append(avg_train_loss)  # Add to training loss list

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)  # Calculate average validation loss
        val_loss_list.append(avg_val_loss)  # Add to validation loss list

        # Print epoch results
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

    # Return epoch-wise training and validation loss
    return train_loss_list, val_loss_list

def extract_features(model, data_loader):
    model.eval()
    features_list, targets_list = [], []
    with torch.no_grad():
        for x, y in data_loader:
            # Extract features from the model
            features = model(x)  # Shape: [batch_size, seq_length]

            # Reshape features to [batch_size, seq_length, 1]
            features_reshaped = features.unsqueeze(2)  # Add a new dimension, resulting in [batch_size, seq_length, 1]

            # Merge features with x along the feature dimension
            combined_features = torch.cat((x, features_reshaped), dim=2)  # Shape: [batch_size, seq_length, feature_dim + 1]

            # Add combined features to the list
            features_list.append(combined_features.numpy())

            # Use only the first day of the target list
            targets_list.append(y[:, 0].numpy())  # Extract the first column

    # Concatenate all features and targets into single arrays
    concatenated_features = np.concatenate(features_list, axis=0)
    concatenated_targets = np.concatenate(targets_list, axis=0)

    # Flatten the last two dimensions of concatenated_features
    flattened_features = concatenated_features.reshape(concatenated_features.shape[0], -1)

    # Print shapes for debugging
    print(f"Original concatenated features shape: {concatenated_features.shape}")
    print(f"Flattened features shape: {flattened_features.shape}")
    print(f"Extracted targets shape: {concatenated_targets.shape}")

    return flattened_features, concatenated_targets

# def extract_features(model, data_loader):
#     model.eval()
#     features_list, targets_list = [], []
#     with torch.no_grad():
#         for x, y in data_loader:
#             features = model(x)
#             features_list.append(features.numpy())
#             targets_list.append(y.numpy())
#     return np.concatenate(features_list, axis=0), np.concatenate(targets_list, axis=0)

def plot_comparison_chart(test_targets, test_preds, plot_dir):
    """
    Plots a comparison chart of predicted vs actual values and saves it to the plot directory.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets.flatten(), test_preds.flatten(), alpha=0.6, label="Predicted vs Actual")
    plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 
             color='red', linestyle='--', label="Ideal Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Comparison of Predicted vs Actual Values")
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_plot_path = f"{plot_dir}/novel_comparison_plot_{timestamp}.png"
    plt.savefig(comparison_plot_path)
    print(f"Comparison chart saved at {comparison_plot_path}")
    # plt.show()
    plt.close()

def plot_learning_curve(train_loss, val_loss, plot_dir):
    """
    Plots the learning curve (training and validation loss over epochs) and saves it to the plot directory.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    learning_curve_path = f"{plot_dir}/novel_learning_curve_{timestamp}.png"
    plt.savefig(learning_curve_path)
    print(f"Learning curve saved at {learning_curve_path}")
    # plt.show()
    plt.close()



# Main Workflow
def do_ensemble_model(filepath, plot_dir):
    """
    Performs ensemble modeling and visualizes results including:
    - R2 score
    - Comparison plots
    - Learning curves
    """
    print("Step 1: Loading and preprocessing data...")
    df = load_data(filepath)
    print(f"Data loaded successfully. Dataset contains {len(df)} records.")

    print("Step 2: Creating datasets and DataLoaders...")
    dataset = TimeSeriesDataset(df, SEQUENCE_LENGTH, FEATURES, TARGET_COLUMN)
    print(f"Dataset created with {len(dataset)} samples.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32*20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32*20, shuffle=False)
    print(f"Train size: {train_size}, Validation size: {val_size}")

    print("Step 3: Initializing the transformer model...")
    transformer = TransformerFeatureExtractor(
        input_dim=len(FEATURES), nhead=4, hidden_dim=64, num_layers=2, output_dim=7
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
    print("Transformer model, loss function, and optimizer initialized.")

    print("Step 4: Training the transformer model...")
    train_loss, val_loss = train_transformer_feature_extractor(transformer, train_loader, 
                                                               val_loader, criterion, optimizer, 
                                                               epochs=200)
    print("Transformer training completed.")

    print("Step 5: Extracting features from the transformer...")
    train_features, train_targets = extract_features(transformer, train_loader)
    test_features, test_targets = extract_features(transformer, val_loader)
    print(f"Feature extraction complete. Train shape: {train_features.shape}, Test shape: {test_features.shape}")

    print("Step 6: Training the XGBoost model on extracted features...")
    print(f"Training data shape for XGBoost - Features: {train_features.shape}, Targets: {train_targets.shape}")
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                                 n_estimators=100, 
                                 max_depth=6, 
                                 learning_rate=0.1)
    xgb_model.fit(train_features, train_targets)
    print("XGBoost training completed.")

    # Step 7: Saving the models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(plot_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save the transformer model
    transformer_save_path = os.path.join(model_save_dir, f"ensemble_transformer_model_{timestamp}.pth")
    torch.save(transformer.state_dict(), transformer_save_path)
    print(f"Transformer model saved at: {transformer_save_path}")
    
    # Save the XGBoost model
    xgb_model_save_path = os.path.join(model_save_dir, f"ensemble_xgboost_model_{timestamp}.json")
    xgb_model.save_model(xgb_model_save_path)
    print(f"XGBoost model saved at: {xgb_model_save_path}")

    print("Step 8: Evaluating the XGBoost model...")
    print(f"Test data shape for XGBoost prediction - Features: {test_features.shape}, Targets: {test_targets.shape}")
    test_preds = xgb_model.predict(test_features)

    print(f"Shape of predictions: {test_preds.shape}")
    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    r2 = r2_score(test_targets, test_preds)
    print(f"RMSE: {rmse:.4f} R2 Score: {r2:.4f}")

    print("Step 9: Generating plots...")
    plot_comparison_chart(test_targets, test_preds, plot_dir)
    plot_learning_curve(train_loss, val_loss, plot_dir)
    print(f"Plots saved in: {plot_dir}")



# Main Script
if __name__ == "__main__":
    filepath = f'{homedir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
    plot_dir = f'{homedir}/../plots'
    os.makedirs(plot_dir, exist_ok=True)
    do_ensemble_model(filepath, plot_dir)


