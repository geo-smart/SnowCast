
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
from snowcast_utils import homedir, work_dir, test_start_date, test_end_date
from convert_results_to_images import convert_csvs_to_images_simple, convert_csv_to_geotiff
import geopandas as gpd

matplotlib.use('Agg')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
SEQUENCE_LENGTH = 7
TARGET_COLUMN = 'swe_value'
PREDICTED_COLUMN = "predicted_swe"

BASE_FEATURES = [
    'SWE', 
    # 'change_in_swe_inch', 'snow_depth', 'air_temperature_observed_f', # remove SNOTEL values
    'precipitation_amount', 'relative_humidity_rmin', 'potential_evapotranspiration',
    'air_temperature_tmmx', 'relative_humidity_rmax', 'mean_vapor_pressure_deficit',
    'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect', 'Curvature',
    'Northness', 'Eastness', 'fsca', 'Slope'
]

# Derived features include lagged values
# FEATURES = BASE_FEATURES + [f"{feature}_lag{i}" for feature in BASE_FEATURES for i in range(1, SEQUENCE_LENGTH + 1)]
FEATURES = BASE_FEATURES

# Data Preprocessing
def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    df = df[df[TARGET_COLUMN] > 0]
    df = df.sort_values(by=["station_name", "date"])
    # df = df.iloc[:10000]  # Limit rows for faster processing
    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, features, target_column):
        """
        Initializes the TimeSeriesDataset object for prediction.

        Args:
            data (numpy array): The data with shape (time_steps, stations, features).
            sequence_length (int): The length of the historical time window (e.g., past 7 days).
        """
        self.data = data  # Shape (7, 462204, 19)
        self.sequence_length = sequence_length
        self.sequences = self.create_sequences()
        self.features = features
        self.target_column = target_column

    def create_sequences(self):
        # reshape the input into (462204, 7, 19)
        reshaped_array = np.transpose(self.data, (1, 0, 2))
        data = np.nan_to_num(reshaped_array, nan=-1)
        return data
        # sequences = []

        # # Extract dimensions
        # num_time_steps, num_stations, num_features = self.data.shape

        # # Ensure there are enough time steps for at least one sequence
        # if num_time_steps < self.sequence_length:
        #     print(f"Warning: Insufficient time steps ({num_time_steps}) for the given sequence length "
        #           f"({self.sequence_length}). No sequences will be created.")
        #     return []

        # # Iterate over each station
        # for station_idx in range(num_stations):
        #     station_data = self.data[:, station_idx, :]  # Shape (time_steps, features)

        #     # Use the last `sequence_length` time steps for prediction
        #     seq = station_data[-self.sequence_length:, :]  # Shape (sequence_length, features)
        #     sequences.append(seq)

        # return np.array(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Fetch the input sequence
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)  # Shape (sequence_length, features)
        return sequence


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


class EnsembleModelPredictor:
    """
    Ensemble Model Predictor for processing the time series.
    """
    def __init__(self, transformer_path, xgb_path, features, sequence_length, target_column, device='cpu'):
        """
        Initializes the EnsembleModelPredictor with paths to the saved models.

        Args:
            transformer_path (str): Path to the saved Transformer model.
            xgb_path (str): Path to the saved XGBoost model.
            features (list): List of feature column names.
            sequence_length (int): Sequence length for the Transformer input.
            target_column (str): Name of the target column.
            device (str): Device to load the Transformer model ('cpu' or 'cuda').
        """
        self.transformer_path = transformer_path
        self.xgb_path = xgb_path
        self.features = features
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.device = device

        # Load models
        print("Loading Transformer model...")
        self.transformer = TransformerFeatureExtractor(
            input_dim=len(features), nhead=4, hidden_dim=64, num_layers=2, output_dim=7
        )
        print("Transformer model configuration:")
        print(f"  Input Dimension: {len(features)}")
        self.transformer.load_state_dict(torch.load(transformer_path, map_location=device))

        # Print the loaded model's details
        # print("Model details after loading weights:")
        # print(self.transformer)

        # Optionally, print all parameter names and their shapes
        # print("\nModel parameters after loading weights:")
        # for name, param in self.transformer.named_parameters():
        #     print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

        # for name, param in self.transformer.named_parameters():
        #     print(f"{name}: min={torch.min(param)}, max={torch.max(param)}, mean={torch.mean(param)}")

        self.transformer.to(device)
        self.transformer.eval()
        print("Transformer model loaded successfully.")

        print("Loading XGBoost model...")
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(xgb_path)
        print("XGBoost model loaded successfully.")

    def preprocess_data(self, data):
        """
        Prepares the input data for prediction.

        Args:
            data (np.array): Input data as a NumPy array.

        Returns:
            DataLoader: DataLoader for the preprocessed data.
        """
        print("Preprocessing input data...")
        dataset = TimeSeriesDataset(data, sequence_length=self.sequence_length, features=self.features, target_column=self.target_column)
        dataloader = DataLoader(dataset, batch_size=640, shuffle=False)
        print(f"Data preprocessed. Total samples: {len(dataset)}")
        return dataloader

    def predict(self, data):
        """
        Performs the two-step prediction using Transformer and XGBoost models.

        Args:
            data (np.array): Input data as a NumPy array.

        Returns:
            np.array: Final predictions for the input data.
        """
        print("data.shape: ", data.shape)
        lat_lon = data[0, :, -2:] # get lat and lon
        print("lat_lon shape: ", lat_lon.shape)
        
        # remove the last three columns
        data = data[:, :, :-2]

        # Step 1: Preprocess the data
        dataloader = self.preprocess_data(data)

        # Step 2: Extract features using the Transformer
        print("Extracting features using the Transformer model...")
        transformer_features = []
        past_sequences = []

        with torch.no_grad():
            for batch in dataloader:
                # print(batch)
                # Check if the batch contains only NaN values
                if torch.isnan(batch).all():
                    print("Error: Batch contains only NaN values. Terminating process.")
                    raise ValueError("Why batch only has nan, no values?")

                # print("batch shape: ", batch.shape)
                batch = batch.to(self.device)
                
                # Extract past sequence features using the Transformer
                future_sequence_features = self.transformer(batch)
                # print("output: ", future_sequence_features)

                if torch.isnan(future_sequence_features).all():
                    print("Error: future_sequence_features contains only NaN values. Terminating process.")
                    raise ValueError("Why does future_sequence_features only have nan values?")
                transformer_features.append(future_sequence_features.cpu().numpy())
                
                # The future sequence (e.g., next time steps in the data) will be used as is
                past_sequences.append(batch.cpu().numpy())

                # break

        # Concatenate features
        transformer_features = np.concatenate(transformer_features, axis=0)
        past_sequences = np.concatenate(past_sequences, axis=0)

        print(past_sequences.shape)
        print(transformer_features.shape)

        # Convert to PyTorch tensors
        transformer_features_tensor = torch.tensor(transformer_features)

        # Add a new dimension (unsqueeze) along the last axis, resulting in [batch_size, seq_length, 1]
        transformer_features_reshaped = transformer_features_tensor.unsqueeze(2)  # Adds a new dimension

        # Merge features with past_sequences along the feature dimension
        # Ensure past_sequences is also a tensor if it's a NumPy array
        past_sequences_tensor = torch.tensor(past_sequences)

        # Concatenate along the feature dimension (dim=2)
        combined_features = torch.cat((past_sequences_tensor, transformer_features_reshaped), dim=2)

        # Print the shape of the combined features
        print(combined_features.shape)

        # Step 3: Combine flattened features for XGBoost input
        print("Combining features for XGBoost input ", combined_features.shape)

        # Flatten the combined features for XGBoost input
        # Ensure combined_features is a PyTorch tensor before reshaping
        flattened_features = combined_features.reshape(combined_features.shape[0], -1)
        print("Flatten combined features: ", flattened_features.shape)

        # Convert the flattened features to NumPy array (required by XGBoost)
        xgb_input = flattened_features.cpu().numpy()  # Use .cpu() in case the tensor is on GPU
        print(f"XGBoost input shape: {xgb_input.shape}")

        # Step 3: Combine past sequence features and future sequence data for XGBoost input
        # xgb_input = np.concatenate([transformer_features_expanded, past_sequences], axis=-1)
        # print(f"XGBoost input shape: {xgb_input.shape}")

        # Step 4: Predict with the XGBoost model
        # Print XGBoost model details
        print("Inspecting the trained XGBoost model...")
        print(f"XGBoost parameters: {self.xgb_model.get_params()}")
        # Retrieve the Booster object
        booster = self.xgb_model.get_booster()
        print("Booster object retrieved.")
        # print(booster.get_dump())

        # Number of features can be inferred from the input or the Booster
        print(f"Booster number of features: {booster.num_features()}")
        # Requires Booster object

        # Step 4: Predict with the XGBoost model
        print("Performing predictions with the XGBoost model...")
        predictions = self.xgb_model.predict(xgb_input)
        print(f"Predictions complete. Output shape: {predictions.shape}")

        
        predictions = predictions.reshape(-1, 1)
        print("predictions.shape: ", predictions.shape)
        final_output = np.concatenate([lat_lon, predictions], axis=-1)
        print(f"Final output shape: {final_output.shape}")
        # print(f"First 5 inputs: {xgb_input[:5]}")
        # print(f"First 5 predictions: {predictions[:5]}")
        return final_output

    def save_predictions(self, predictions, output_path):
        """
        Saves the predictions to a CSV file.

        Args:
            predictions (np.array): Predicted values.
            output_path (str): Path to save the predictions.
        """
        print(f"Saving predictions to {output_path}...")
        # the predictions contains all 7 future days. Only output 1 day
        df = pd.DataFrame(predictions, columns=["Latitude", "Longitude", PREDICTED_COLUMN])
        print(df.describe())
        df.to_csv(output_path, index=False)
        print("Predictions saved successfully.")


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def prepare_input_numpy_for_date(target_date="2024-11-20", output_path: str = None, force=False):
    try:
        # Debug: Start of function
        print("Starting the process to prepare input NumPy array...")
        print(f"Target date: {target_date}")

        # Save as .npy file
        if output_path is None:
            output_path = f"{homedir}/bttf_output/combined_array_{target_date}.npy"
        output_path_dir = os.path.dirname(output_path)
        os.makedirs(output_path_dir, exist_ok=True)

        if os.path.exists(output_path) and not force:
            print(f"File exists: {output_path}")
            return output_path

        # Parse the target date
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")

        # Generate the past 7 days including the target date
        past_dates = [(target_date_obj - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        print(f"Past 7 days: {past_dates}")

        # File path and pattern
        base_path = f"{work_dir}/"  # Replace with actual path
        file_pattern = "testing_all_ready_{}.csv"

        # Initialize list to hold data for each day
        daily_data = []
        all_columns = []

        for date in reversed(past_dates):
            file_name = file_pattern.format(date)
            file_path = os.path.join(base_path, file_name)

            if not os.path.exists(file_path):
                print(f"Warning: File for date {date} not found: {file_path}")
                continue

            print(f"Loading file: {file_path}")
            daily_df = pd.read_csv(file_path)
            
            # Ensure the 'date' column is consistent and matches the file's date
            daily_df['date'] = date
            print("original df columns: ", daily_df.columns)
            all_columns = daily_df.columns
            # Append to the list
            daily_data.append(daily_df)

        if len(daily_data) < 7:
            raise ValueError(f"Insufficient data: Found data for {len(daily_data)} days, but 7 days are required.")

        # Combine all data into a single DataFrame
        combined_df = pd.concat(daily_data, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}", combined_df.columns)

        column_mapping = {
            'AMSR_SWE': 'SWE',
            'tmmx': 'air_temperature_observed_f',  # this column is duplicated with tmmx, why?
            'pr': 'precipitation_amount',
            'rmin': 'relative_humidity_rmin',
            'etr': 'potential_evapotranspiration',
            'tmmx': 'air_temperature_tmmx',
            'rmax': 'relative_humidity_rmax',
            'vpd': 'mean_vapor_pressure_deficit',
            'tmmn': 'air_temperature_tmmn',
            'vs': 'wind_speed',
        }
        combined_df = combined_df.rename(columns=column_mapping)

        combined_df = combined_df.fillna(-1)
        print("Assign -1 to fsca and SWE column where values are >100..")
        combined_df.loc[combined_df['fsca'] > 100, 'fsca'] = -1
        combined_df.loc[combined_df['SWE'] > 100, 'SWE'] = -1

        # Filter necessary columns
        # variable_columns = ['Latitude', 'Longitude', 'tmmn', 'etr', 
        #                     'rmin', 'vpd', 'tmmx', 'rmax', 
        #                     'vs', 'pr', 'Elevation', 'Slope', 'Aspect', 'Curvature', 
        #                     'Northness', 'Eastness', 'AMSR_SWE', 'cumulative_AMSR_SWE']
        variable_columns = BASE_FEATURES
        print("Only retain ", variable_columns)
        all_variable_columns = variable_columns + ['date', 'Latitude','Longitude',] # must save the date and lat and lon into the numpy
        final_variable_columns = variable_columns + ['Latitude','Longitude',]
        combined_df = combined_df[all_variable_columns]
        print("Filtered columns to include necessary variables.")

        # Sort data by Latitude, Longitude, and date
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(by=['Latitude', 'Longitude', 'date'])

        # Group data by grid points and reshape
        grouped = combined_df.groupby(['Latitude', 'Longitude'])
        numpy_array_list = []

        print(combined_df.head())
        print("the 19 columns: ", combined_df.columns)

        for (lat, lon), group in grouped:
            group_sorted = group.sort_values(by='date')
            if len(group_sorted) == 7:
                numpy_array_list.append(group_sorted[final_variable_columns].to_numpy())

        if not numpy_array_list:
            raise ValueError("No grids found with exactly 7 days of data.")

        # Stack arrays into final shape
        final_array = np.stack(numpy_array_list, axis=1)  # Shape: (7, grids, variables)
        print(f"Final array shape: {final_array.shape}")
        np.save(output_path, final_array)
        print(f"Saved NumPy array at: {output_path}")

        return final_array

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def load_and_analyze_numpy(file_path):
    try:
        # Load the saved NumPy array
        print(f"Loading NumPy array from: {file_path}")
        data = np.load(file_path)

        print(f"Array shape: {data.shape} (days, grids, variables)")
        column_names = BASE_FEATURES + ["Latitude", "Longitude"]

        # Validate the number of columns matches the array shape
        if len(column_names) != data.shape[2]:
            print("Warning: Number of column names does not match the number of variables in the data.")


        # Compute statistics
        stats = {
            'mean': np.mean(data, axis=(0, 1)),
            'min': np.min(data, axis=(0, 1)),
            'max': np.max(data, axis=(0, 1)),
            'std': np.std(data, axis=(0, 1)),
        }

        # Print statistics for each variable
        for i, col_name in enumerate(column_names):
            variable_data = data[:, :, i]
            mean_val = np.nanmean(variable_data)  # Use nanmean to avoid NaNs in statistics
            min_val = np.nanmin(variable_data)
            max_val = np.nanmax(variable_data)
            std_val = np.nanstd(variable_data)
            print(f"Variable {i+1} ({col_name}): Mean={mean_val:.3f}, Min={min_val:.3f}, Max={max_val:.3f}, Std={std_val:.3f}")

    except Exception as e:
        print(f"An error occurred while loading or analyzing the array: {e}")

def do_prediction(target_date="2024-11-20", output_prediction_path:str = None, force=False):
    # Paths to saved models
    transformer_model_path = f"{homedir}/../models/ensemble_transformer_model_20250105_091448.pth"
    xgb_model_path = f"{homedir}/../models/ensemble_xgboost_model_20250105_091448.json"

    print("Model used for prediction: ", transformer_model_path, xgb_model_path)

# 50 epoch
#     Transformer model saved at: /media/volume1/swe/data//../plots/models/ensemble_transformer_model_20250104_220829.pth
# XGBoost model saved at: /media/volume1/swe/data//../plots/models/ensemble_xgboost_model_20250104_220829.json

# 200 epoch
# Transformer model saved at: /media/volume1/swe/data//../plots/models/ensemble_transformer_model_20250105_044829.pth
# XGBoost model saved at: /media/volume1/swe/data//../plots/models/ensemble_xgboost_model_20250105_044829.json

# 500 epoch
# ransformer model saved at: /media/volume1/swe/data//../plots/models/ensemble_transformer_model_20250105_091448.pth
# XGBoost model saved at: /media/volume1/swe/data//../plots/models/ensemble_xgboost_model_20250105_091448.json

    # Path to input CSV and output predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # target_date = "2024-11-20"
    # input_csv = "/groups/ESS3/zsun/swe/data/testing_ready_input/input_2024-11-20.csv"
    input_np_path = f"{homedir}/bttf_output/combined_array_{target_date}.npy"

    if output_prediction_path is None:
        output_prediction_path = f"{homedir}/bttf_output/transformer_xgb_predictions_{target_date}.csv"

    if os.path.exists(output_prediction_path) and not force:
        print(f"File exists: {output_prediction_path}")
        return output_prediction_path

    input_np_path_dir = os.path.dirname(input_np_path)
    os.makedirs(input_np_path_dir, exist_ok=True)
    output_np_path_dir = os.path.dirname(output_prediction_path)
    os.makedirs(output_np_path_dir, exist_ok=True)

    # Initialize the predictor
    predictor = EnsembleModelPredictor(
        transformer_path=transformer_model_path,
        xgb_path=xgb_model_path,
        features=FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        target_column=TARGET_COLUMN
    )

    # Run predictions
    data = np.load(input_np_path)  # Example of loading your data

    predictions = predictor.predict(data)

    # Save predictions
    predictor.save_predictions(predictions, output_prediction_path,)


if __name__ == "__main__":
    target_date = "2025-01-01"
    bttf_npy_file = f"{homedir}/bttf_output/combined_array_{target_date}.npy"
    prepare_input_numpy_for_date(target_date=target_date, output_path = bttf_npy_file, force=True)
    load_and_analyze_numpy(file_path = bttf_npy_file)
    # columns
#     'Latitude', 'Longitude', 'tmmn', 'etr', 'rmin', 'vpd', 'tmmx', 'rmax',
# >        'vs', 'pr', 'Elevation', 'Slope', 'Aspect', 'Aspect', 'Curvature',
# >        'Northness', 'Eastness', 'AMSR_SWE', 'cumulative_AMSR_SWE', 'date']
    predicted_csv = f"{homedir}/bttf_output/transformer_xgb_predictions_{target_date}.csv"
    do_prediction(
        target_date=target_date, output_prediction_path=predicted_csv, force=True
    )
    # convert_csvs_to_images_simple(target_date = target_date)
    convert_csvs_to_images_simple(
        target_date=target_date, 
        column_name = "predicted_swe", 
        test_csv = predicted_csv,
        res_png_path = f"{homedir}/bttf_output/{target_date}.png"
    );
    convert_csv_to_geotiff(
        target_date, 
        test_csv=predicted_csv, 
        target_geotiff_file=f"{homedir}/bttf_output/swe_predicted_{target_date}.tif"
    )
    


