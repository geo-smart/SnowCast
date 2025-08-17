from base_nn_hole import BaseNNHole
import torch
import torch.nn as nn
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import psutil
from snowcast_utils import model_dir, data_dir, plot_dir, test_start_date, test_end_date
from datetime import datetime, timedelta
import time
import subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from base_hole import BaseHole
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(out_features)
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class SNODAS_NewResNet_Model(nn.Module):
    def __init__(self, input_dim=8):
        super(SNODAS_NewResNet_Model, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),  # Wider first layer
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(256)
        )

        self.hidden_layers = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 128)  # Reduce back down
        )

        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1)
        )

        self.apply(self.init_weights)  # Apply better weight initialization

    def forward(self, X):
        X = self.input_layer(X)
        X = self.hidden_layers(X)
        return self.output_layer(X)

    @staticmethod
    def init_weights(m):
        """ Apply Kaiming initialization for better gradient flow """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)
    
    def load_model(self, model_path):
        """Loads the model weights from a file."""
        try:
            # Load the state dict from the file
            state_dict = torch.load(model_path)
            # Load the state dict into the model
            self.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

class SNODASNewResNetHole(BaseNNHole):

    def normalize(self, X):
        return X

    def preprocessing(self, verbose=False, semimonth_only = False):
        """
        Preprocesses the data for training and testing.

        Returns:
            None
        """
        
        # List all CSV files that end with -01.csv or -15.csv
        if semimonth_only:
            files = [
                os.path.join(self.SNODAS_CSV_FOLDER, f)
                for f in os.listdir(self.SNODAS_CSV_FOLDER)
                if f.endswith('.csv') and ('-01_snodas' in f or '-15_snodas' in f)
            ]
        else:
            files = [
                os.path.join(self.SNODAS_CSV_FOLDER, f)
                for f in os.listdir(self.SNODAS_CSV_FOLDER)
                if f.endswith('.csv')
            ]

        # Shuffle the files to ensure randomness
        random.shuffle(files)

        # Split the files: 80% for training and 20% for evaluation
        train_size = int(self.train_ratio * len(files))
        test_size = int(self.test_ratio * len(files))
        self.train_files = files[:train_size]
        self.eval_files = files[-test_size:]

        if verbose:
            print("Train files:", self.train_files)
            print("Test files:", self.eval_files)
    
    def encode_features(self, df):
        """
        Converts spatial and temporal features into an 8D representation:
        - Uses sin transformations for Latitude and Longitude (to maintain continuity)
        - Uses sin/cos transformations for periodic variables (year, month, day_of_year)
        """
        # Encode Latitude and Longitude using sin transformation for generalization
        # Normalize Year (assuming range ~[2000, 2100])
        df['year_enc'] = (df['year'] - 2000) / 100  

        # Normalize Month (1-12 → scaled between 0 and 1)
        df['month_enc'] = (df['month'] - 1) / 11  

        # Normalize Day of Year (1-365 → scaled between 0 and 1)
        df['day_enc'] = (df['day_of_year'] - 1) / 364  

        # Select final 8D input features
        return df[['Latitude', 'Longitude', 'year', 'month', 'day_of_year', 'year_enc', 'month_enc', 'day_enc']].values

    def load_batch(self, files):
        batch_data = []
        
        for file in files:
            if not os.path.exists(file):
                continue
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'])
            batch_data.append(df)
        
        batch_df = pd.concat(batch_data, ignore_index=True)

        # Extract date components
        batch_df['day_of_year'] = batch_df['date'].dt.dayofyear
        batch_df['month'] = batch_df['date'].dt.month
        batch_df['year'] = batch_df['date'].dt.year

        # Transform into 8D input features
        X = self.encode_features(batch_df)
        X *= 100

        # Log-transform the target variable
        y = np.log10(batch_df['snodas'] + 1)  # +1 to avoid log(0)
        
        return X, y

    def predict_swe_map(
        self, 
        target_date: str, 
        model_path: str = None, 
        batch_size: int = -1
    ):
        if model_path is None:
            model_path = self.model_path
        self.model = self.load_model(model_path)

        if batch_size == -1:
            batch_size = self.batch_size

        model_file_name = os.path.basename(self.model_path)

        print(f"Model loaded successfully. Running evaluation on {self.device}...")
        print("reading: ", self.LAT_LON_CSV_PATH, " - batch_size: ", batch_size)
        
        mapper_df = pd.read_csv(self.LAT_LON_CSV_PATH, chunksize=batch_size)  # Read in chunks

        # Step 2: Parse the date string to extract year, month, and day_of_year
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        year, month, day_of_year = date_obj.year, date_obj.month, date_obj.timetuple().tm_yday

        print(year, month, day_of_year)

        result_df_list = []  # List to store processed batches

        # Process in batches using tqdm for progress tracking
        for batch in tqdm(mapper_df, desc="Processing batches", unit="batch"):
            if 'Latitude' not in batch.columns or 'Longitude' not in batch.columns:
                raise ValueError("The CSV file must contain 'Latitude' and 'Longitude' columns.")

            # Step 3: Add time-related columns to the batch
            batch['year'], batch['month'], batch['day_of_year'] = year, month, day_of_year

            batch = batch[['Latitude', 'Longitude', 'day_of_year', 'month', 'year']]
            X = self.encode_features(batch)

            # Step 4: Extract the features
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            # Run inference
            with torch.no_grad():
                y = self.model(X_tensor).cpu().numpy()

            # Flatten if necessary
            if y.ndim > 1:
                y = y.flatten()

            # Apply transformation
            batch['SWE_predicted'] = (10 ** y) - 1  # Reverse log10 scaling
            if batch.empty or len(batch) == 0:
                raise ValueError("Encountered an empty batch. Please check the data source.")
            
            # Correctly print the first and last Latitude/Longitude
            print(f"Batch first row: {batch.iloc[0]['Latitude']}, {batch.iloc[0]['Longitude']}")
            print(f"Batch last row: {batch.iloc[-1]['Latitude']}, {batch.iloc[-1]['Longitude']}")
            
            result_df_list.append(batch)

        # Concatenate all processed batches
        final_mapper_df = pd.concat(result_df_list, ignore_index=True)
        if final_mapper_df.isna().any().any():
            raise ValueError("DataFrame contains NaN values. Please check the input data.")

        self.save_png(final_mapper_df, target_date, model_file_name)

        return final_mapper_df


    def get_model(self):
        model = SNODAS_NewResNet_Model().to(self.device)
        # Calculate the total number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")
        return model



if __name__ == "__main__":

    # Define start and end date for training
    start_date_str = '2024-10-01'
    end_date_str = '2025-01-15'
    
    hole_model = SNODASNewResNetHole(
        start_date_str = test_start_date,
        end_date_str = test_end_date,
        batch_size = 310000,
        epochs = 5,
        train_ratio = 0.8,
        test_ratio = 0.05,
        val_ratio = 0.01,
        normalization = True,
        retrain = True,
        # model_path = "/home/chetana/models//SNODASNewResNetHole_e1_nTrue_20250202143527.model",
        base_model_path = "/home/chetana/models//SNODASNewResNetHole_e1_nTrue_20250202143527.model"
    )
    
    hole_model.preprocessing(verbose=True, semimonth_only = False)
    hole_model.train()
    hole_model.save()
    hole_model.evaluate()
    hole_model.predict("2024-12-10")
    hole_model.predict("2024-11-10")
    hole_model.predict("2024-10-10")
    hole_model.predict("2024-09-10")
    hole_model.predict("2024-08-10")
    hole_model.predict("2024-07-10")
    hole_model.predict("2024-06-10")
    hole_model.predict("2024-05-10")
    hole_model.predict("2024-04-10")
    hole_model.predict("2024-03-10")
    hole_model.predict("2024-02-10")
    hole_model.predict("2024-01-10")


