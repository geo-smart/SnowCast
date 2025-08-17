import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
from snowcast_utils import data_dir, work_dir, output_dir
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import psutil
from snowcast_utils import model_dir, data_dir, test_start_date, test_end_date
import random
import subprocess
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings
import matplotlib.pyplot as plt
from snodas_dnn_new import SNODAS_DNN_Model, SNODASDNNHole

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet.callbacks")


# Function to get system memory usage
def get_system_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Return in MB

# Function to get GPU memory usage
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)  # Return in MB
    return 0

def print_memory_usage(step_description):
    system_memory = get_system_memory()
    gpu_memory = get_gpu_memory()
    print(f"Memory usage ({step_description}):")
    print(f"System Memory: {system_memory:.2f} MB")
    print(f"GPU Memory: {gpu_memory:.2f} MB")
    print("-" * 40)

def get_first_available_device():
    """
    Returns the first available GPU device based on utilization or falls back to CPU.
    
    Returns:
        torch.device: The device to use for computations.
    """
    try:
        # Fetch the list of GPUs and their UUIDs
        gpu_list = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            universal_newlines=True
        )
        
        gpu_id_to_uuid = {}
        for line in gpu_list.strip().split("\n"):
            if line:
                gpu_id, gpu_uuid = line.split(", ")
                gpu_id_to_uuid[gpu_uuid] = int(gpu_id)
        
        # Fetch processes using GPUs
        gpu_usage = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            universal_newlines=True
        )
        
        # Track GPUs in use
        used_gpus = set(gpu_usage.strip().split("\n"))
        
        # Find the first GPU not in use
        for gpu_uuid, gpu_id in gpu_id_to_uuid.items():
            if gpu_uuid not in used_gpus:
                print(f"Using GPU: {gpu_id} - {torch.cuda.get_device_properties(gpu_id).name}")
                return torch.device(f"cuda:{gpu_id}")
        
        print("No free GPUs available. Using CPU.")
    except Exception as e:
        print(f"Error checking GPU usage: {e}. Falling back to CPU.")
    
    return torch.device("cpu")

device = get_first_available_device()

def get_model(device):
    model = SNODAS_DNN_Model(norm = True).to(device)
    # Calculate the total number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {total_params}")
    return model

def add_snodas_mask_column(
    model_path: str, 
    training_csv, 
    new_training_csv,
    lat_col_name = "lat", 
    lon_col_name = "lon", 
    date_col_name = "date",
    overwrite: bool = True,
):

    
    import pandas as pd
    import numpy as np
    from scipy.spatial import cKDTree

    # Load training data
    training_df = pd.read_csv(training_csv)
    training_df.rename(columns={
        lat_col_name: 'lat', 
        lon_col_name: 'lon',
        date_col_name: 'date'
    }, inplace=True)
    print(training_df.head())

    snodas_mask_csv = f"{output_dir}/SNODASDNNHole_e10_nTrue_nn_swe_predicted_2024-12-10.csv"
    if os.path.exists(snodas_mask_csv) and not overwrite:
        print(f"{new_training_csv} exists and won't overwrite. Skip..")
        predicted_df = pd.read_csv(snodas_mask_csv)
    else:
        # Generate predictions
        hole_model = SNODASDNNHole(
            start_date_str=test_start_date,
            end_date_str=test_end_date,
            batch_size=310000,
            epochs=10,
            train_ratio=0.8,
            test_ratio=0.2,
            val_ratio=0.01,
            normalization=True,
            retrain=True,
            model_path="/home/chetana/models//SNODASDNNHole_e10_nTrue_20250202144843.model",
        )

        predicted_df = hole_model.predict("2024-12-10", save_csv=True, tiff=True)

    print("predicted_df.columns: ", predicted_df.columns)
    predicted_df.rename(columns={
        'Latitude': 'lat', 
        'Longitude': 'lon'
    }, inplace=True)

    # Ensure required columns exist
    if not {'lat', 'lon'}.issubset(training_df.columns) or not {'lat', 'lon'}.issubset(predicted_df.columns):
        raise ValueError("Both training_df and predicted_df must contain 'lat' and 'lon' columns.")

    # Build KDTree for fast nearest-neighbor lookup
    tree = cKDTree(predicted_df[['lat', 'lon']].values)

    # Query KDTree for the closest SWE prediction for each row in training_df
    distances, indices = tree.query(training_df[['lat', 'lon']].values, k=1)

    # Add the nearest SWE value to training_df
    training_df['snodas_mask'] = predicted_df.iloc[indices]['SWE_predicted'].values  # Adjust 'SWE' to the actual column name

    # Save or return updated DataFrame
    training_df.to_csv(new_training_csv, index=False)

    # Print summary
    print("New csv with snodas_mask column is saved to ", new_training_csv)
    

if __name__ == "__main__":
    # Call the function with appropriate paths
    # WARNING: This is not right, because batch matters. The start of each batch will matter. The correct way is to first generate all years of data, and then use the output to generate the training.csv.
    training_csv = f"{work_dir}/all_points_final_merged_training.csv"
    new_training_csv = f"{work_dir}/all_points_final_merged_training_snodas_mask_resnet_all_batch.csv"
    model_path = f"/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101055710.model"
    add_snodas_mask_column(
        model_path, 
        training_csv, 
        new_training_csv, 
        lat_col_name = "lat", 
        lon_col_name = "lon", 
        date_col_name = "date",
        overwrite= False,
    )
