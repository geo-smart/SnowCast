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

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

        # Shortcut connection
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out)) + identity
        return self.relu(out)

class SNODAS_NewResNet_Model(nn.Module):
    def __init__(self, input_dim=8):
        super(SNODAS_NewResNet_Model, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.hidden_layers = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 128)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, X):
        X = self.input_layer(X)
        X = self.hidden_layers(X)
        return self.output_layer(X)
    
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
        df['lat_enc'] = np.sin(np.pi * df['Latitude'] / 180)
        df['lon_enc'] = np.sin(np.pi * df['Longitude'] / 180)

        # Encode Year (assuming range ~[2000, 2100])
        df['year_sin'] = np.sin(2 * np.pi * (df['year'] - 2000) / 100)
        df['year_cos'] = np.cos(2 * np.pi * (df['year'] - 2000) / 100)

        # Encode Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Encode Day of Year (365-day cycle)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Select final 8D input features
        return df[['lat_enc', 'lon_enc', 'year_sin', 'year_cos', 
                'month_sin', 'month_cos', 'day_sin', 'day_cos']].values

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

        # Log-transform the target variable
        y = np.log10(batch_df['snodas'] + 1)  # +1 to avoid log(0)
        
        return X, y

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
        epochs = 1,
        train_ratio = 0.8,
        test_ratio = 0.2,
        val_ratio = 0.01,
        normalization = True,
        retrain = False,
        # model_path = "/home/chetana/models//SNODASNewResNetHole_e10_nTrue_20253101221511.model",
        # base_model_path = "/home/chetana/models//SNODASNewResNetHole_e10_nTrue_20253101221511.model"
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



