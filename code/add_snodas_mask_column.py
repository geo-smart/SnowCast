import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
from snowcast_utils import data_dir, work_dir
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

def add_snodas_mask_column_for_training(
    model_path: str, 
    training_csv, 
    new_training_csv,
    lat_col_name = "lat", 
    lon_col_name = "lon", 
    date_col_name = "date"
):
    # the training must be added from AI-derived swe maps. This is because if the batch size changes, the results will change. Must maintain the same input and batch size as the training. 
    training_df = pd.read_csv(training_csv)
    print(training_df.columns)
    unique_dates = training_df[["date"]].drop_duplicates()
    print("unique dates: ", len(unique_dates))

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
        model_path = "/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101055710.model",
        # base_model_path = "/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101055710.model"
    )
    
    # hole_model.preprocessing(verbose=True, semimonth_only = False)
    # hole_model.train()
    # hole_model.save()
    # hole_model.evaluate()
    for date in unique_dates["date"]:
        print("Generating for ", date)
        df = hole_model.predict(date, save_csv=False, tiff=False, skip_exists=False)

        print(df.describe())
        return

    

    

# Main function for prediction on the large CSV
def add_snodas_mask_column_for_testing(
    model_path: str, 
    testing_csv, 
    new_testing_csv,
    lat_col_name = "lat", 
    lon_col_name = "lon", 
    date_col_name = "date"
):
    print(f"Loading model from: {model_path}")
    device = get_first_available_device()
    model = get_model(device)
    model.load_model(model_path)
    
    print("Model loaded successfully. Preparing the dataset...")
    
    # Read large CSV efficiently in chunks
    chunk_size = 310000  # Set a chunk size that your system can handle
    # this chunk size must match the batch size when training the model. Different chunk size will cause problems. Cannot change this!!! Be aware.
    lat_lon_csv_path = testing_csv
    
    # Placeholder for predictions (to add them to the dataframe)
    predictions_list = []
    
    model.eval()
    for chunk in pd.read_csv(lat_lon_csv_path, chunksize=chunk_size):
        # print(chunk.columns)
        # Extract required columns
        current_chunk = chunk[[date_col_name, lat_col_name, lon_col_name]].copy()
        
        # Parse the date and extract year, month, day of year
        print("Parsing date and extracting features...")
        current_chunk[date_col_name] = pd.to_datetime(current_chunk[date_col_name], format="%Y-%m-%d")
        current_chunk['year'] = current_chunk[date_col_name].dt.year
        current_chunk['month'] = current_chunk[date_col_name].dt.month
        current_chunk['day_of_year'] = current_chunk[date_col_name].dt.dayofyear
        
        # Extract features (Latitude, Longitude, day_of_year, month, year)
        X = current_chunk[[lat_col_name, lon_col_name, 'day_of_year', 'month', 'year']].values

        # y = model.predict(X)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = model(X).detach().cpu().numpy()

        # Flatten y if it has an extra dimension
        if y.ndim > 1:
            y = y.flatten()

        # Convert to a pandas Series for convenience
        y_series = pd.Series(y)

        snodas_mask = (10 ** y) - 1  # Reverse log10 scaling, should be 0.1

        # Add predictions to the chunk
        chunk['snodas_mask'] = snodas_mask
        
        # Append the chunk with predictions to the list
        predictions_list.append(chunk)

        print(f"Processed {len(chunk)} rows, predictions added.")
    
    # Concatenate all the chunks with predictions
    final_df = pd.concat(predictions_list, ignore_index=True)
    
    # Save the final dataframe with predictions
    output_file = new_testing_csv
    final_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Call the function with appropriate paths
    # WARNING: This is not right, because batch matters. The start of each batch will matter. The correct way is to first generate all years of data, and then use the output to generate the training.csv.
    training_csv = f"{work_dir}/all_points_final_merged_training.csv"
    new_training_csv = f"{work_dir}/all_points_final_merged_training_snodas_mask_resnet.csv"
    model_path = f"/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101055710.model"
    add_snodas_mask_column_for_training(
        model_path, 
        training_csv, 
        new_training_csv, 
        lat_col_name = "lat", 
        lon_col_name = "lon", 
        date_col_name = "date"
    )

