import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import psutil
from snowcast_utils import model_dir, data_dir, plot_dir
import random
import subprocess
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings
import matplotlib.pyplot as plt

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

# Replace TransformerModel with TabNet model
class TabNet(nn.Module):
    def __init__(self,input_dim,output_dim,n_d=64,n_a=64,
n_shared=2,n_ind=2,n_steps=5,relax=1.2,vbs=128):
        super().__init__()
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim,2*(n_d+n_a)))
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a,2*(n_d+n_a)))
        else:
            self.shared=None
        self.first_step = FeatureTransformer(input_dim,n_d+n_a,self.shared,n_ind) 
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(input_dim,n_d,n_a,self.shared,n_ind,relax,vbs))
        self.fc = nn.Linear(n_d,output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.n_d = n_d

    def forward(self,x):
        x = self.bn(x)
        x_a = self.first_step(x)[:,self.n_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te,l = step(x,x_a,priors)
            out += F.relu(x_te[:,:self.n_d])
            x_a = x_te[:,self.n_d:]
            sparse_loss += l
        return self.fc(out),sparse_loss

# Load and preprocess data for a specific batch (5 days)
def load_batch(files):
    batch_data = []
    
    # Iterate over the files and filter data by date range
    for file in files:
        if not os.path.exists(file):
            continue
        
        df = pd.read_csv(file)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter data for the date range of this batch (start_date to end_date)
        batch_data.append(df)
    
    # Concatenate all data for the batch
    batch_df = pd.concat(batch_data)
    
    # Feature engineering: Extract day of year and month
    batch_df['day_of_year'] = batch_df['date'].dt.dayofyear
    batch_df['month'] = batch_df['date'].dt.month
    batch_df['year'] = batch_df['date'].dt.year
    
    # scale snodas to deal with bias
    batch_df['snodas'] = np.log10(batch_df['snodas'] + 1)  # +1 to avoid log(0)

    # Features (Latitude, Longitude, day_of_year, month)
    X = batch_df[['Latitude', 'Longitude', 'day_of_year', 'month', 'year']].values
    
    # Target: SWE (snodas)
    y = batch_df['snodas'].values
    
    return X, y

def new_batch_generator(files, start_date, end_date, batch_size):
    num_files = len(files)
    current_index = 0
    X_accumulated = []
    y_accumulated = []

    while current_index < num_files:
        current_file = files[current_index]
        
        # Load the data for this file
        X_file, y_file = load_batch([current_file])

        # Accumulate rows from this file
        X_accumulated.extend(X_file)
        y_accumulated.extend(y_file)

        # If the accumulated data size exceeds or reaches batch_size, yield batches
        while len(X_accumulated) >= batch_size:
            X_chunk = X_accumulated[:batch_size]
            y_chunk = y_accumulated[:batch_size]
            
            # Remove yielded data from accumulation
            X_accumulated = X_accumulated[batch_size:]
            y_accumulated = y_accumulated[batch_size:]

            # Convert lists to numpy arrays and reshape y_chunk
            X_chunk = np.array(X_chunk)
            y_chunk = np.array(y_chunk).reshape(-1, 1)  # Ensure y_chunk is a 2D array
            
            yield X_chunk, y_chunk

        current_index += 1

    # Yield any remaining data if it's less than batch_size
    if X_accumulated:
        X_remaining = np.array(X_accumulated)
        y_remaining = np.array(y_accumulated).reshape(-1, 1)
        yield X_remaining, y_remaining

def get_model(device):
    # model = TabNet(
    #     input_dim=5, output_dim=1
    # )
    # model = TabNetRegressor()
    model = TabNetRegressor(
        input_dim=5,
        output_dim=1,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.5,
        lambda_sparse=1e-5,
        device_name = str(device)
    )
    return model

def train_model_in_batches(
    files, start_date, end_date, epochs=100, batch_size=50000,
    model_save_path=f"{model_dir}/snodas_tabnet_model.pth"
):
    device = get_first_available_device()
    model = get_model(device)

    # params_dict = {k: torch.tensor(v) if isinstance(v, (int, float)) else v for k, v in model.optimizer_params.items()}
    # optimizer = optim.Adam(params_dict, lr=0.001)
    criterion = nn.MSELoss()
    print("TabNet is set, ready to train")

    # Send model to GPUs
    
    # model = model.to(device)
    # model.device_name = device
    
    print("Starting model training...")
    print_memory_usage("start training")
    
    big_batch_num = 0
    epoch_loss = 0
    # for epoch in range(epochs):
        # print(f"Big wrap epoch {epoch} / {epochs}")
    for X_batch, y_batch in new_batch_generator(files, start_date, end_date, batch_size):
        big_batch_num += 1
        print("processing ", big_batch_num, " - ", len(X_batch))
        # Split the batch into training and validation sets (e.g., 80% train, 20% validation)
        split_idx = int(0.8 * len(X_batch))  # 80% for training, 20% for validation
        
        X_train, y_train = X_batch[:split_idx], y_batch[:split_idx]
        X_val, y_val = X_batch[split_idx:], y_batch[split_idx:]

        # Reshape targets to 2D for regression
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        # Training the model using the fit method
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["rmse"],
            max_epochs=epochs,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=1,
            drop_last=False,
            warm_start=True,  # Ensures continued training from the last state
            loss_fn=torch.nn.MSELoss()  # Use MSELoss for regression tasks
        )

        # After training, make predictions on the validation set
        # y_val_pred = model.predict(X_val)

        # Get the minimum value of the validation loss (logits_ll)
        # score = np.min(model.history["val_logits_ll"])

        # print(f"Validation score: {score}")

        # Compute the loss
        # Ensure that y_val_pred and y_val are tensors
        # y_val_pred = torch.tensor(y_val_pred, dtype=torch.float32)
        # y_val = torch.tensor(y_val, dtype=torch.float32)
        # loss = criterion(y_val_pred, y_val)
        # epoch_loss += loss.item()

        # Backward pass and optimization
        # loss.backward()
        # optimizer.step()

    # print(f"Epoch {epoch+1} completed.")

    print("Model training completed.")
    # Save the trained model to a file
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}.")
    return model


# Evaluate the model
def evaluate_model(model_file_path, files, start_date, end_date, batch_size=10000):
    print("Loading model from:", model_file_path)
    device = get_first_available_device()
    model = get_model(device)
    model.load_model(f"{model_file_path}.zip")

    print(f"Model loaded successfully. Running evaluation on {device}...")

    all_y_true = []
    all_y_pred = []

    # Start batch processing
    batch_count = 0
    for X_batch, y_batch in new_batch_generator(files, start_date, end_date, batch_size):
        y_pred = model.predict(X_batch)  # Get model predictions
        all_y_true.extend(y_batch)
        all_y_pred.extend(y_pred)

    # Calculate evaluation metrics
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluation Results: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

def predict_swe_map(data_folder: str, target_date: str, model_path: str):
    print("Loading model from:", model_path)
    device = get_first_available_device()
    model = get_model(device)
    model.load_model(model_path)

    model_file_name = os.path.basename(model_path)

    print(f"Model loaded successfully. Running evaluation on {device}...")
    lat_lon_csv_path = "/home/chetana/data/snodas/2025-01-19_snodas_output.csv"
    mapper_df = pd.read_csv(lat_lon_csv_path)
    # Ensure required columns exist
    if 'Latitude' not in mapper_df.columns or 'Longitude' not in mapper_df.columns:
        raise ValueError("The CSV file must contain 'Latitude' and 'Longitude' columns.")
    
    # Step 2: Parse the date string to extract year, month, and day_of_year
    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    year = date_obj.year
    month = date_obj.month
    day_of_year = date_obj.timetuple().tm_yday  # Get day of the year

    print(year, month, day_of_year)
    
    # Step 3: Add the time-related columns to the DataFrame
    mapper_df['year'] = year
    mapper_df['month'] = month
    mapper_df['day_of_year'] = day_of_year
    
    # Step 4: Extract the features (Latitude, Longitude, day_of_year, month, year)
    X = mapper_df[['Latitude', 'Longitude', 'day_of_year', 'month', 'year']].values
    
    print(X)

    y = model.predict(X)

    # Flatten y if it has an extra dimension
    if y.ndim > 1:
        y = y.flatten()

    # Convert to a pandas Series for convenience
    y_series = pd.Series(y)

    # Print basic statistics
    print("Statistics of predictions (y):")
    print(f"Mean: {y_series.mean()}")
    print(f"Median: {y_series.median()}")
    print(f"Standard Deviation: {y_series.std()}")
    print(f"Minimum: {y_series.min()}")
    print(f"Maximum: {y_series.max()}")

    swe_predictions = (10 ** y) - 0.1  # Reverse log10 scaling

    # Add predictions to the DataFrame
    mapper_df['SWE_predicted'] = swe_predictions

    print(mapper_df['SWE_predicted'].describe())

    mapper_df = mapper_df[(mapper_df['SWE_predicted'] >= 0) & (mapper_df['SWE_predicted'] <= 3000)]
    
    # Plotting the SWE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        mapper_df['Longitude'], 
        mapper_df['Latitude'], 
        c=mapper_df['SWE_predicted'], 
        cmap='viridis', 
        s=10
    )
    plt.colorbar(scatter, label="SWE (mm)")
    plt.title(f"Predicted SWE Map for {target_date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/swe_map_{target_date}_{model_file_name}.png")
    plt.show()

    print(f"SWE map saved to {plot_dir}/swe_map_{target_date}_{model_file_name}.png")


# Main function to execute the training process
def main(data_folder, start_date_str, end_date_str, model_path):
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # List all CSV files in the data folder
    # Loop through dates from start_date to end_date (inclusive)
    current_date = start_date
    while current_date <= end_date:
        # Convert the current date back to a string (if needed)
        current_date_str = current_date.strftime("%Y-%m-%d")
        
        # Perform your processing for the current date
        print(f"Processing data for date: {current_date_str}")
        
        # Example: Placeholder for data processing function
        predict_swe_map(data_folder, current_date_str, model_path)
        
        # Increment the date by 1 day
        current_date += timedelta(days=1)

if __name__ == "__main__":
    # Folder containing the data files
    data_folder = f"{data_dir}/snodas/csv/"

    # Define start and end date for training
    start_date_str = '2025-01-29'
    end_date_str = '2025-01-29'
    # epoch 2
    # model_path = f"/home/chetana/models/snodas_tabnet_20250122_074129.pth.zip" 
    # epoch 10
    # model_path = f"/home/chetana/models/snodas_tabnet_20250122_075325.pth.zip"
    # epoch 1000 on 1 file
    # model_path = f"/home/chetana/models/snodas_tabnet_20250128_132543.pth.zip"
    # epoch 100 on 5 files
    model_path = f"/home/chetana/models/snodas_tabnet_20250128_145240.pth.zip"

    # Run the training process
    model = main(data_folder, start_date_str, end_date_str, model_path)

