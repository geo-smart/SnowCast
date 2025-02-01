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

class SNODAS_DNN_Model(nn.Module):
    def __init__(self, norm):
        self.norm = norm
        super(SNODAS_DNN_Model, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.hidden_layers = nn.Sequential(
            self._residual_block(128, 256),
            self._residual_block(256, 256),
            self._residual_block(256, 256),
            self._residual_block(256, 256),
            self._residual_block(256, 128)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _residual_block(self, in_features, out_features):
        """Creates a residual block for faster training"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, X):
        X = self.input_layer(X)
        X = self.hidden_layers(X) + X  # Residual Connection
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

class SNODASDNNHole(BaseNNHole):

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

    def get_model(self):
        model = SNODAS_DNN_Model(norm = self.normalization).to(self.device)
        # Calculate the total number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")
        return model



if __name__ == "__main__":

    # Define start and end date for training
    start_date_str = '2024-10-01'
    end_date_str = '2025-01-15'
    
    hole_model = SNODASDNNHole(
        start_date_str = test_start_date,
        end_date_str = test_end_date,
        batch_size = 310000,
        epochs = 10,
        train_ratio = 0.8,
        test_ratio = 0.2,
        val_ratio = 0.01,
        normalization = True,
        retrain = True,
        # model_path = "/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101221511.model",
        base_model_path = "/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101221511.model"
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



