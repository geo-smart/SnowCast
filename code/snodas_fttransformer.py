# This doesn't work, only gives noise

import torch
import torch.nn as nn
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import psutil
from snowcast_utils import model_dir, data_dir, plot_dir, output_dir, test_start_date, test_end_date
from datetime import datetime, timedelta
import time
import subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from base_nn_hole import BaseNNHole

# Function to define the FT-Transformer model
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, 
        embed_dim=16, num_heads=4, num_layers=2
    ):
        super(FTTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Define the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2
        )
        
        # Use TransformerEncoder with batch_first=True for (batch_size, seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output regressor layer
        self.regressor = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # Input embedding: [batch_size, sequence_length, embed_dim]
        x = self.embedding(x)
        # print("after embedding: ", x.shape)
        
        # Add a sequence length dimension of 1, resulting in shape [batch_size, 1, embed_dim]
        x = x.unsqueeze(1)  # Adds a dimension for seq_len, so now it's [batch_size, 1, embed_dim]
        
        # Transformer expects inputs with shape (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        
        # Pass through the transformer encoder
        x = self.transformer(x)
        # print("after transformer: ", x.shape)
        
        # Pooling the sequence output (mean over the sequence length)
        x = torch.mean(x, dim=0)  # Mean pooling over the sequence length
        # print("after mean: ", x.shape)

        # Remove sequence length dimension
        x = x.squeeze(0)  # [batch_size, embed_dim]
        
        # Regress to the final output
        return self.regressor(x)

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


class SNODAS_FTTransformer_Hole(BaseNNHole):

    def get_model(self):
        model = FTTransformer(
            input_dim=5, output_dim=1, embed_dim=64, num_heads=4, num_layers=5
        ).to(self.device)
        # Calculate the total number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")
        return model


if __name__ == "__main__":
    
    hole_model = SNODAS_FTTransformer_Hole(
        start_date_str = test_start_date,
        end_date_str = test_end_date,
        batch_size = 300000,
        epochs = 50,
        train_ratio = 0.1,
        test_ratio = 0.1,
        val_ratio = 0.2,
        retrain = False,
        normalization = False,
    )
    
    hole_model.preprocessing(verbose=True)
    hole_model.train()
    hole_model.save()
    hole_model.evaluate()
    hole_model.predict("2025-01-29", save_csv=True, tiff=True)

