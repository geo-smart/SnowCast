
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
from sklearn.ensemble import ExtraTreesRegressor

from bttf_train import BASE_FEATURES, SEQUENCE_LENGTH, TARGET_COLUMN, TimeSeriesNetCDFDataset, TransformerFeatureExtractor, train_transformer_feature_extractor, extract_features, plot_comparison_chart, plot_learning_curve, TARGET_COLUMN

matplotlib.use('Agg')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Derived features include lagged values
# FEATURES = BASE_FEATURES + [f"{feature}_lag{i}" for feature in BASE_FEATURES for i in range(1, SEQUENCE_LENGTH + 1)]
FEATURES = BASE_FEATURES

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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Fetch the input sequence
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)  # Shape (sequence_length, features)
        return sequence


class EnsembleModelPredictor:
    """
    Ensemble Model Predictor for processing the time series.
    """
    def __init__(self, transformer_path, visionary_path, features, sequence_length, target_column, device='cpu'):
        """
        Initializes the EnsembleModelPredictor with paths to the saved models.

        Args:
            transformer_path (str): Path to the saved Transformer model.
            visionary_path (str): Path to the saved XGBoost model.
            features (list): List of feature column names.
            sequence_length (int): Sequence length for the Transformer input.
            target_column (str): Name of the target column.
            device (str): Device to load the Transformer model ('cpu' or 'cuda').
        """
        self.transformer_path = transformer_path
        self.visionary_path = visionary_path
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

        print("Loading Visionary model...")

        # ET model loading
        # self.visionary_model = ExtraTreesRegressor(
        #     random_state=42,
        #     n_jobs=-1             # use all CPUs
        # )
        # import joblib
        # self.visionary_model = joblib.load(visionary_path)
        # print("ExtraTrees model loaded from ", visionary_path)

        # XGBoost loading
        self.visionary_model = xgb.XGBRegressor()
        self.visionary_model.load_model(visionary_path)
        print("Visionary model loaded successfully.")

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

        # save the first day prediction of transformer
        transformer_predictions = transformer_features[:, 0].reshape(-1, 1)
        print("transformer first day predictions.shape: ", transformer_predictions.shape)
        transformer_output = np.concatenate([lat_lon, transformer_predictions], axis=-1)
        print(f"transformer output shape: {transformer_output.shape}")


        # Convert to PyTorch tensors
        transformer_features_tensor = torch.tensor(transformer_features)

        # Add a new dimension (unsqueeze) along the last axis, resulting in [batch_size, seq_length, 1]
        transformer_features_reshaped = transformer_features_tensor.unsqueeze(2)  # Adds a new dimension

        # Merge features with past_sequences along the feature dimension
        # Ensure past_sequences is also a tensor if it's a NumPy array
        past_sequences_tensor = torch.tensor(past_sequences)

        # Concatenate along the feature dimension (dim=2)
        combined_features = torch.cat((past_sequences_tensor, transformer_features_reshaped), dim=2) # for meta-learning
        # combined_features = past_sequences_tensor # for pure ET

        # Print the shape of the combined features
        print(combined_features.shape)

        # Step 3: Combine flattened features for Visionary input
        print("Combining features for Visionary input ", combined_features.shape)

        # Flatten the combined features for Visionary input
        # Ensure combined_features is a PyTorch tensor before reshaping
        flattened_features = combined_features.reshape(combined_features.shape[0], -1)
        print("Flatten combined features: ", flattened_features.shape)

        # Convert the flattened features to NumPy array (required by Visionary)
        xgb_input = flattened_features.cpu().numpy()  # Use .cpu() in case the tensor is on GPU
        print(f"Visionary input shape: {xgb_input.shape}")

        # Step 3: Combine past sequence features and future sequence data for Visionary input
        # xgb_input = np.concatenate([transformer_features_expanded, past_sequences], axis=-1)
        # print(f"Visionary input shape: {xgb_input.shape}")

        # Step 4: Predict with the Visionary model
        # Print Visionary model details
        print("Inspecting the trained Visionary model...")
        print(f"Visionary parameters: {self.visionary_model.get_params()}")
        # Retrieve the Booster object
        # booster = self.visionary_model.get_booster()
        # print("Booster object retrieved.")
        # print(booster.get_dump())

        # Number of features can be inferred from the input or the Booster
        # print(f"Booster number of features: {booster.num_features()}")
        # Requires Booster object

        # Step 4: Predict with the Visionary model
        print("Performing predictions with the Visionary model...")
        predictions = self.visionary_model.predict(xgb_input)
        print(f"Predictions complete. Output shape: {predictions.shape}")

        # if predicted value is minus, assign 0
        # merged_df.loc[merged_df['predicted_swe'] < 0, 'predicted_swe'] = 0
        
        # # if GridMET has no data, don't output
        # merged_df.loc[merged_df['air_temperature_tmmx']==0, 
        #             'predicted_swe'] = 0

        # # if there are water body, don't output
        # merged_df.loc[merged_df['lc_prop3'] == 3, 'predicted_swe'] = 0
        # merged_df.loc[merged_df['lc_prop3'] == 255, 'predicted_swe'] = 0
        # merged_df.loc[merged_df['lc_prop3'] == 27, 'predicted_swe'] = 0

        
        predictions = predictions.reshape(-1, 1)
        print("predictions.shape: ", predictions.shape)
        final_output = np.concatenate([lat_lon, predictions], axis=-1)
        print(f"Final output shape: {final_output.shape}")
        # print(f"First 5 inputs: {xgb_input[:5]}")
        # print(f"First 5 predictions: {predictions[:5]}")
        return final_output, transformer_output

    def save_predictions(self, predictions, output_path):
        """
        Saves the predictions to a CSV file.

        Args:
            predictions (np.array): Predicted values.
            output_path (str): Path to save the predictions.
        """
        print(f"Saving predictions to {output_path}...")
        # the predictions contains all 7 future days. Only output 1 day
        df = pd.DataFrame(predictions, columns=["Latitude", "Longitude", TARGET_COLUMN])
        print(df.describe())

        # clean up and remove the no snow areas
        mask_csv_path = os.path.join(work_dir, "testing_all_ready", "testing_all_ready_2025-08-31.csv_snodas_mask.csv")
        # mask_df = pd.read_csv(mask_csv_path)
        mask_df = pd.read_csv(
            mask_csv_path,
            usecols=["lat", "lon", "tmmx", "lc_prop3"]  # replace with your 4 column names
        )

        mask_df = mask_df.rename(columns={
            "lat": "Latitude",
            "lon": "Longitude"
        })

        merged = pd.merge(df, mask_df, on=["Latitude", "Longitude"], how="inner")

        # if predicted value is minus, assign 0
        merged.loc[merged[TARGET_COLUMN] < 0, TARGET_COLUMN] = 0
        
        # if GridMET has no data, don't output
        merged.loc[merged['tmmx']==0, TARGET_COLUMN] = 0

        # if there are water body, don't output
        merged.loc[merged['lc_prop3'] == 3, TARGET_COLUMN] = 0
        # df.loc[mask_df['lc_prop3'] == 255, TARGET_COLUMN] = 0
        # df.loc[mask_df['lc_prop3'] == 27, TARGET_COLUMN] = 0

        merged[["Latitude", "Longitude", TARGET_COLUMN]].to_csv(output_path, index=False)
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
            file_path = os.path.join(base_path, "testing_all_ready", file_name)

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

def do_prediction(
    transformer_model_path,
    visionary_model_path,
    target_date="2024-11-20", 
    output_prediction_path:str = None, 
    output_transformer_prediction_path: str = None,
    force=False
):

    print("Model used for prediction: ", transformer_model_path, visionary_model_path)

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
    input_np_path = f"{homedir}/data/bttf_output/combined_array_{target_date}.npy"

    if output_prediction_path is None:
        output_prediction_path = f"{homedir}/data/bttf_output/transformer_xgb_predictions_{target_date}.csv"

    if output_transformer_prediction_path is None:
        output_transformer_prediction_path = f"{homedir}/data/bttf_output/transformer_predictions_{target_date}.csv"

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
        visionary_path=visionary_model_path,
        features=FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        target_column=TARGET_COLUMN
    )

    # Run predictions
    data = np.load(input_np_path)  # Example of loading your data

    predictions, transformer_predictions = predictor.predict(data)

    # Clean up and Save predictions
    predictor.save_predictions(predictions, output_prediction_path,)
    predictor.save_predictions(transformer_predictions, output_transformer_prediction_path,)


if __name__ == "__main__":
    # Paths to saved models
    # transformer_model_path = f"{homedir}/models/ensemble_transformer_model_20250906_025046.pth"
    # transformer_model_path = f"{homedir}/models/ensemble_transformer_model_20250913_072844.pth" # 500 epochs no physics
    # xgb_model_path = f"{homedir}/models/ensemble_xgboost_model_20250906_025046.json"\
    # et_model_path = f"{homedir}/models/ensemble_extratrees_model_20250910_025302.joblib"
    # et_model_path = f"{homedir}/models/ensemble_extratrees_model_20250910_025302.joblib"
    # et_model_path = f"{homedir}/models/ensemble_extratrees_model_20250911_040229.joblib"
    # xgb_model_path = f"{homedir}/models/ensemble_xgboost_model_20250913_072844.json"

    # 10 epoch elevation no imputation
    # time_stamp = "20250914_020446"

    # 10 epoch 80% zero dropped
    time_stamp = "20250914_050029"

    transformer_model_path = f"{homedir}/models/ensemble_transformer_model_{time_stamp}.pth"
    xgb_model_path = f"{homedir}/models/ensemble_xgboost_model_{time_stamp}.json"


    target_date = "2025-01-17"
    bttf_npy_file = f"{homedir}/data/bttf_output/combined_array_{target_date}.npy"
    prepare_input_numpy_for_date(target_date=target_date, output_path = bttf_npy_file, force=False)
    load_and_analyze_numpy(file_path = bttf_npy_file)
    # columns
#     'Latitude', 'Longitude', 'tmmn', 'etr', 'rmin', 'vpd', 'tmmx', 'rmax',
# >        'vs', 'pr', 'Elevation', 'Slope', 'Aspect', 'Aspect', 'Curvature',
# >        'Northness', 'Eastness', 'AMSR_SWE', 'cumulative_AMSR_SWE', 'date']
    predicted_csv = f"{homedir}/data/bttf_output/transformer_xgb_predictions_{target_date}.csv"
    transformer_predicted_csv = f"{homedir}/data/bttf_output/transformer_predictions_{target_date}.csv"

    do_prediction(
        transformer_model_path,
        xgb_model_path,
        target_date=target_date, 
        output_prediction_path=predicted_csv, 
        output_transformer_prediction_path = transformer_predicted_csv,
        force=True
    )
    # convert_csvs_to_images_simple(target_date = target_date)
    convert_csvs_to_images_simple(
        target_date=target_date, 
        column_name = TARGET_COLUMN, 
        test_csv = predicted_csv,
        res_png_path = f"{homedir}/data/bttf_output/{target_date}_ensemble_xgboost_model_{time_stamp}.png"
    );
    convert_csvs_to_images_simple(
        target_date=target_date, 
        column_name = TARGET_COLUMN, 
        test_csv = transformer_predicted_csv,
        res_png_path = f"{homedir}/data/bttf_output/{target_date}_transformer_model_{time_stamp}.png"
    );

    convert_csv_to_geotiff(
        target_date, 
        test_csv=predicted_csv, 
        target_geotiff_file=f"{homedir}/data/bttf_output/swe_predicted_{target_date}_ensemble_xgboost_model_{time_stamp}.tif",
        target_column=TARGET_COLUMN
    )

    convert_csv_to_geotiff(
        target_date, 
        test_csv=transformer_predicted_csv, 
        target_geotiff_file=f"{homedir}/data/bttf_output/swe_predicted_{target_date}_transformer_model_{time_stamp}.tif",
        target_column=TARGET_COLUMN
    )
    

