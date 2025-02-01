import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import psutil
from snowcast_utils import model_dir, data_dir
import random
import subprocess

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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Linear layers to project input to d_model
        self.src_linear = nn.Linear(input_dim, d_model)
        
        # Define the transformer layer (using only the encoder here)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Output layer to predict SWE value
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        """
        Forward pass of the Transformer model with shape printing.

        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # print(f"Input shape: {src.shape}")  # Shape: (batch_size, input_dim)

        # Project inputs to d_model
        src = self.src_linear(src)  # Shape: (batch_size, d_model)
        # print(f"After src_linear shape: {src.shape}")

        # Transformer expects (seq_len, batch, d_model), so reshape inputs accordingly
        # src = src.unsqueeze(0)  # Add sequence dimension, shape: (1, batch_size, d_model)
        # print(f"Before transformer shape: {src.shape}")

        # Pass through the transformer
        output = self.transformer(src)  # Shape: (1, batch_size, d_model)
        # print(f"After transformer shape: {output.shape}")

        # Use the output of the first sequence token for prediction
        output = output.squeeze(0)  # Remove sequence dimension, shape: (batch_size, d_model)
        # print(f"After squeeze shape: {output.shape}")

        # Predict the SWE value
        output = self.fc_out(output)  # Shape: (batch_size, output_dim)
        # print(f"Output shape: {output.shape}")

        return output



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
    
    # print(batch_df["snodas"].describe())

    # scale snodas to deal with bias
    batch_df['snodas'] = np.log10(batch_df['snodas'] + 1)  # +1 to avoid log(0)
    # print(batch_df["snodas"].describe())

    # Features (Latitude, Longitude, day_of_year, month)
    X = batch_df[['Latitude', 'Longitude', 'day_of_year', 'month', 'year']].values
    
    # Target: SWE (snodas)
    y = batch_df['snodas'].values
    
    return X, y

def new_batch_generator(files, start_date, end_date, batch_size):
    """
    Generate batches of data from files.
    
    Args:
        files (list): List of file paths containing the dataset.
        start_date (str): Start date for filtering data.
        end_date (str): End date for filtering data.
        batch_size (int): Number of rows per batch.
    
    Yields:
        (X_batch, y_batch): Tuple of feature and target batches.
    """
    # Sort the files by date (assuming file names are in the format 'YYYY-MM-DD')
    files.sort()  # Adjust if necessary to ensure files are in chronological order

    # Filter files based on the provided date range
    # filtered_files = []
    # for file in files:
    #     # Extract date from the file name (assuming the date is part of the file name)
    #     # Adjust this if your file name format differs
    #     file_date_str = os.path.basename(file).split('_')[0]  # Example: "2025-01-01_data.csv"
    #     file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()

    #     # Add file to list if it's within the date range
    #     if start_date <= file_date <= end_date:
    #         filtered_files.append(file)
    
    num_files = len(files)
    current_index = 0

    # Loop through the filtered files and create batches
    while current_index < num_files:

        current_file = files[current_index]
        # print("current_file = ", current_file)
        
        # Load the data for this batch (you need to implement load_batch to handle files)
        X_batch, y_batch = load_batch([current_file])

        num_samples = X_batch.shape[0]
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # Slice the batch into smaller chunks
            X_chunk = X_batch[start_idx:end_idx]
            y_chunk = y_batch[start_idx:end_idx]

            # Yield the batch
            yield X_chunk, y_chunk

        current_index += 1

def get_model():
    model = TransformerModel(
        input_dim=5, d_model=32, num_heads=4, num_layers=2, output_dim=1
    )
    return model

def train_model_in_batches(
    files, start_date, end_date, epochs=100, batch_size=50000,
    model_save_path=f"{model_dir}/snodas_transformer_model.pth"
):
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print("Transformer is set, ready to train")

    # Send model to GPUs
    device = get_first_available_device()
    model = model.to(device)
    
    print("Starting model training...")
    print_memory_usage("start training")

    for epoch in range(epochs):
        print(f"Training epoch {epoch} / {epochs}")
        model.train()  # Set model to training mode
        
        big_batch_num = 0
        epoch_loss = 0
        for X_batch, y_batch in new_batch_generator(files, start_date, end_date, batch_size):
            big_batch_num += 1
            # print(f"Epoch {epoch+1}/{epochs}, File Batch {big_batch_num}...")
            
            # print_memory_usage(f"Training epoch {epoch} file {big_batch_num}")
            
            # Convert to torch tensors
            X_batch = torch.tensor(X_batch, dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

            # Ensure chunks are on the correct device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Add sequence dimension and permute for transformer
            src = X_batch.unsqueeze(1)  # Add sequence length dimension
            src = src.permute(1, 0, 2)  # Change to (seq_len, batch_size, input_dim)

            # Forward pass for each smaller chunk
            optimizer.zero_grad()
            y_pred = model(src)  # Pass the source to the model
            # print("y_pred.shape: ", y_pred.shape)

            # Compute the loss
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed. Loss: {epoch_loss / big_batch_num:.4f}")

    print("Model training completed.")
    # Save the trained model to a file
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}.")
    return model


# Evaluate the model
def evaluate_model(model_file_path, files, start_date, end_date, batch_size=10000):
    """
    Evaluate the model on the given data files between start_date and end_date.
    
    Args:
        model_file_path (str): Path to the saved model file.
        files (list): List of file paths containing the dataset.
        start_date (str): Start date for filtering data.
        end_date (str): End date for filtering data.
        batch_size (int): Number of rows per batch.
    """
    print("Loading model from:", model_file_path)
    
    model = get_model()
    model.load_state_dict(torch.load(model_file_path))

    # Move model to GPU if available
    device = get_first_available_device()
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    print(f"Model loaded successfully. Running evaluation on {device}...")

    all_y_true = []
    all_y_pred = []

    # Start batch processing
    batch_count = 0
    for X_batch, y_batch in new_batch_generator(files, start_date, end_date, batch_size):
        # print(f"Processing batch {batch_count + 1}...")
        
        # Convert to torch tensors and move to the appropriate device
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1).to(device)

        # print(f"Batch {batch_count + 1} - X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")

        # Add sequence dimensions for transformer input
        if X_batch.dim() == 2:
            X_batch = X_batch.unsqueeze(1)  # Add a sequence length dimension: (batch_size, 1, input_dim)
        
        src = X_batch.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)

        with torch.no_grad():
            # Predict (no target sequence provided during evaluation)
            y_pred = model(src).squeeze(1)  # Shape: (batch_size, output_dim)
        
        # Check the shape of the prediction
        # print(f"Batch {batch_count + 1} - y_pred shape: {y_pred.shape}")

        # Collect predictions and true values
        all_y_true.append(y_batch.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())

        # print(f"Batch {batch_count + 1} processed.")
        
        # Free up GPU memory
        torch.cuda.empty_cache()

        # Increment batch counter
        batch_count += 1

    # Flatten arrays
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    print("all_y_true.shape = ", all_y_true.shape)
    print("all_y_pred.shape = ", all_y_pred.shape)

    # Ensure that the shapes match
    if all_y_true.shape[0] != all_y_pred.shape[0]:
        print(f"Warning: Mismatch in number of samples: {all_y_true.shape[0]} vs {all_y_pred.shape[0]}")

    # Calculate evaluation metrics
    rmse = mean_squared_error(all_y_true, all_y_pred, squared=False)
    r2 = r2_score(all_y_true, all_y_pred)

    print(f"Evaluation completed.")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


# Main function to execute the training process
def main(data_folder, start_date_str, end_date_str):
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # List all CSV files in the data folder
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    # Shuffle the files to ensure randomness
    random.shuffle(files)

    # Split the files: 80% for training and 20% for evaluation
    train_files = files[:int(0.8 * len(files))]
    eval_files = files[int(0.8 * len(files)):]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file_path = f"{model_dir}/snodas_transformer_{timestamp}.pth"

    batch_size = 200000
    epochs = 500
    
    # Train the model using batches of 5 days
    train_model_in_batches(
        train_files, start_date, end_date, epochs=epochs,
        batch_size=batch_size,
        model_save_path = model_file_path,
    )

    # Evaluate model on the last batch of data
    evaluate_model(
        model_file_path, eval_files, end_date - timedelta(days=4), end_date, batch_size=batch_size,
    )
    
    return model_file_path

if __name__ == "__main__":
    # Folder containing the data files
    data_folder = f"{data_dir}/snodas/csv/"

    # Define start and end date for training
    start_date_str = '2024-10-01'
    end_date_str = '2025-01-15'

    # Run the training process
    model = main(data_folder, start_date_str, end_date_str)

