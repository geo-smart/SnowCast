
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
from base_hole import BaseHole
import shutil
from sklearn.utils import shuffle

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata


class SNODASModel(nn.Module):
    def __init__(self):
        super(SNODASModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 128),  # Increased capacity
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            # Repeated block for consistency
            self._build_block(256),
            self._build_block(256),
            self._build_block(256),
            self._build_block(256),
            self._build_block(256),
            self._build_block(256),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def _build_block(self, input_size):
        """Helper function to define a common block for reuse"""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)

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
        for line in gpu_list.splitlines():
            if line:
                gpu_id, gpu_uuid = line.split(", ")
                gpu_id_to_uuid[gpu_uuid] = int(gpu_id)
        
        # Fetch processes using GPUs
        gpu_usage = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            universal_newlines=True
        )
        
        # Track GPUs in use
        used_gpus = set(gpu_usage.splitlines())
        
        # Find the first GPU not in use
        for gpu_uuid, gpu_id in gpu_id_to_uuid.items():
            if gpu_uuid not in used_gpus:
                print(f"Using GPU: {gpu_id} - {torch.cuda.get_device_properties(gpu_id).name}")
                return torch.device(f"cuda:{gpu_id}")
        
        print("No free GPUs available. Using CPU.")
    except Exception as e:
        print(f"Error checking GPU usage: {e}. Falling back to CPU.")
    
    return torch.device("cpu")

def csv_to_geotiff(csv_file, output_tiff, value_column="SWE_predicted", resolution_km=1):
    """
    Convert CSV with latitude, longitude, and value data into a GeoTIFF raster with a 1 km resolution.

    Parameters:
    - csv_file (str): Path to the input CSV file.
    - output_tiff (str): Path to save the output GeoTIFF.
    - value_column (str): Column in the CSV file to rasterize.
    - resolution_km (float): Desired grid resolution in kilometers (default: 1 km).

    Returns:
    - None
    """

    # Load CSV
    df = pd.read_csv(csv_file)

    # Drop rows with invalid lat/lon
    df = df[(df["Latitude"] != -999) & (df["Longitude"] != -999)]

    # Extract coordinate and value columns
    lat, lon, values = df["Latitude"].values, df["Longitude"].values, df[value_column].values

    # Calculate degree resolution based on 1 km spacing
    lat_res = resolution_km / 111.32  # Approximate for latitude
    lon_res = resolution_km / (111.32 * np.cos(np.radians(lat.mean())))  # Adjust for longitude

    # Define grid extent
    min_lon, max_lon = lon.min(), lon.max()
    min_lat, max_lat = lat.min(), lat.max()

    # Create grid with 1 km resolution
    grid_x, grid_y = np.meshgrid(
        np.arange(min_lon, max_lon, lon_res),
        np.arange(min_lat, max_lat, lat_res)
    )

    # Interpolate scattered points to a grid
    grid_z = griddata((lon, lat), values, (grid_x, grid_y), method="linear")

    # ✅ Flip the grid along the latitude axis
    grid_z = np.flipud(grid_z)

    # Define transform (fix top-left origin)
    transform = from_origin(min_lon, max_lat, lon_res, lat_res)


    if os.path.exists(output_tiff):
        os.remove(output_tiff)
    
    # Write to GeoTIFF
    with rasterio.open(
        output_tiff, "w",
        driver="GTiff",
        height=grid_z.shape[0],
        width=grid_z.shape[1],
        count=1,
        dtype=grid_z.dtype,
        crs="EPSG:4326",  # WGS84 projection
        transform=transform
    ) as dst:
        dst.write(grid_z, 1)

    print(f"GeoTIFF saved to {output_tiff}")

class BaseNNHole(BaseHole):
    '''
    Base class for snowcast_wormhole neural network predictors.

    Attributes:
        all_ready_file (str): The path to the CSV file containing the data for training.
        classifier: The machine learning model used for prediction.
        holename (str): The name of the wormhole class.
        train_x (numpy.ndarray): The training input data.
        train_y (numpy.ndarray): The training target data.
        test_x (numpy.ndarray): The testing input data.
        test_y (numpy.ndarray): The testing target data.
        test_y_results (numpy.ndarray): The predicted results on the test data.
        save_file (str): The path to save the trained model.
    '''

    
    # Folder containing the data files
    SNODAS_CSV_FOLDER = f"{data_dir}/snodas/csv/"

    LAT_LON_CSV_PATH = "/home/chetana/data/snodas/2025-01-19_snodas_output.csv"

    def __init__(
        self, 
        start_date_str, end_date_str, 
        model_path=None, 
        base_model_path=None,
        epochs=5, 
        batch_size=100000,
        train_ratio = 0.8,
        val_ratio = 0.2,
        test_ratio = 0.2,
        retrain: bool = False, 
        normalization: bool = False,
    ):
        '''
        Initializes a new instance of the BaseHole class.
        '''
        self.holename = self.__class__.__name__ 

        self.train_ratio = 0.8

        # Convert date strings to datetime objects
        self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        self.device = get_first_available_device()
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio


        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.test_y_results = None

        self.train_files = []
        self.retrain = retrain
        self.normalization = normalization
        self.base_model_path = base_model_path
        
        self.batch_size = batch_size
        self.save_file = None
        if model_path is None:
            now = datetime.now()
            date_time = now.strftime("%Y%d%m%H%M%S")
            model_path = f"{model_dir}/{self.holename}_e{epochs}_n{self.normalization}_{date_time}.model"
        else:
            print("Provided model and will continue to use: ", model_path)
        self.model_path = model_path
        self.latest_model_file = f"{model_dir}/{self.holename}_e{epochs}_n{self.normalization}_latest.model"


        self.model = self.get_model()
        if self.retrain:
            print("Current base model: ", self.base_model_path)
            self.load_model(model_path = self.base_model_path)

    
    def save(self):
        '''
        Save the trained model to a joblib file with a timestamp.

        Returns:
            None
        '''
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

        shutil.copy(self.model_path, self.latest_model_file)
        print(f"a copy of the model is saved to {self.latest_model_file}")
  
    def preprocessing(self, verbose=False):
        '''
        Preprocesses the data for training and testing.

        Returns:
            None
        '''
        
        # List all CSV files in the data folder
        files = [
            os.path.join(self.SNODAS_CSV_FOLDER, f) 
            for f in os.listdir(self.SNODAS_CSV_FOLDER) if f.endswith('.csv')
        ]
        # Shuffle the files to ensure randomness
        # random.shuffle(files)
        # Split the files: 80% for training and 20% for evaluation
        self.train_files = files[:int(self.train_ratio * len(files))]
        self.eval_files = files[-int(self.test_ratio * len(files)):]
        if verbose:
            print("Train files: ", self.train_files)
            print("Test files: ", self.eval_files)
  
    def train(self):
        '''
        Trains the machine learning model.

        Returns:
            None
        '''
        # Train the model using batches of 5 days
        self.train_model_in_batches(
            self.train_files, 
            self.start_date, 
            self.end_date, 
            epochs=self.epochs,
            batch_size=self.batch_size,
            model_save_path = self.model_path,
            val_ratio = self.val_ratio
        )
  
    # Evaluate the model
    def evaluate_model(
        self, model_file_path, files, start_date, end_date, batch_size=10000
    ):
        self.model = self.load_model()

        print(f"Model loaded successfully. Running evaluation on {self.device}...")

        all_y_true = []
        all_y_pred = []

        # Start batch processing
        self.model.eval()
        batch_count = 0
        for X_batch, y_batch in self.new_batch_generator(files, batch_size):
            # X_batch = feature_scaler.fit_transform(X_batch)
            X_batch = self.normalize(X_batch)
            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
            y_pred = self.model(X_batch_tensor).detach().cpu().numpy()  # Get model predictions
            all_y_true.extend(y_batch)
            all_y_pred.extend(y_pred)

        # Calculate evaluation metrics
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"Evaluation Results: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

    def evaluate(self):
        '''
        Evaluates the performance of the machine learning model.

        Returns:
            None
        '''
        # Evaluate model on the last batch of data
        self.evaluate_model(
            self.model_path, 
            self.eval_files, 
            self.end_date - timedelta(days=4), 
            self.end_date,
            batch_size=self.batch_size,
        )
  
    def load_model(self, model_path = None):
        self.model = self.get_model()
        if model_path is None:
            model_path = self.model_path
        print("Loading model from:", model_path)
        self.model.load_model(f"{model_path}")
        return self.model

    # Load and preprocess data for a specific batch
    def load_batch(self, files):
        batch_data = []
        for file in files:
            if not os.path.exists(file):
                continue
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'])
            batch_data.append(df)
        batch_df = pd.concat(batch_data)

        batch_df['day_of_year'] = batch_df['date'].dt.dayofyear
        batch_df['month'] = batch_df['date'].dt.month
        batch_df['year'] = batch_df['date'].dt.year

        X = batch_df[['Latitude', 'Longitude', 'day_of_year', 'month', 'year']].values
        
        y = np.log10(batch_df['snodas'] + 1)  # +1 to avoid log(0)
        return X, y

    def new_batch_generator(self, files, batch_size):
        num_files = len(files)
        current_index = 0
        X_accumulated, y_accumulated = [], []

        while current_index < num_files:
            current_file = files[current_index]
            X_file, y_file = self.load_batch([current_file])
            X_accumulated.extend(X_file)
            y_accumulated.extend(y_file)

            while len(X_accumulated) >= batch_size:
                X_chunk = np.array(X_accumulated[:batch_size])
                y_chunk = np.array(y_accumulated[:batch_size]).reshape(-1, 1)
                X_accumulated, y_accumulated = X_accumulated[batch_size:], y_accumulated[batch_size:]
                
                yield X_chunk, y_chunk

            current_index += 1

        if X_accumulated:
            yield np.array(X_accumulated), np.array(y_accumulated).reshape(-1, 1)

    def get_model(self):
        model = SNODASModel().to(self.device)
        # Calculate the total number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")
        return model

    # Define RMSE calculation function
    def calculate_rmse(self, predictions, targets):
        return torch.sqrt(((predictions - targets) ** 2).mean())

    def normalize(self, X):
        if self.normalization:
            print("do norm")
            pass

        return X

    def train_model_in_batches(
        self, 
        files, 
        start_date, end_date, 
        epochs=100, batch_size=50000, 
        model_save_path="ft_transformer_model.pth", 
        val_ratio=0.2
    ):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()  # Record start time for the epoch
            epoch_loss = 0
            epoch_rmse = 0  # To track RMSE for the whole epoch
            batch_num = 0

            # Training loop
            for X_batch, y_batch in self.new_batch_generator(files, self.batch_size):
                batch_num += 1
                # X and y must be shuffled together to ensure the row correspondance
                # X_batch, y_batch = shuffle(X_batch, y_batch, random_state=42)

                # shuffle the X_batch indices
                indices = np.random.permutation(X_batch.shape[0])  # Generate shuffled indices
                X_batch = X_batch[indices]  # Shuffle X
                y_batch = y_batch[indices]  # Shuffle y

                X_batch = self.normalize(X_batch)
                
                # Split batch into training and validation sets based on val_ratio
                split_idx = int((1 - val_ratio) * X_batch.shape[0])  # 80% for training
                X_train_batch, X_val_batch = X_batch[:split_idx], X_batch[split_idx:]
                y_train_batch, y_val_batch = y_batch[:split_idx], y_batch[split_idx:]

                # Convert to torch tensors and move to device
                X_train_batch = torch.tensor(X_train_batch, dtype=torch.float32).to(self.device)
                y_train_batch = torch.tensor(y_train_batch, dtype=torch.float32).to(self.device)
                X_val_batch = torch.tensor(X_val_batch, dtype=torch.float32).to(self.device)
                y_val_batch = torch.tensor(y_val_batch, dtype=torch.float32).to(self.device)

                # Training step
                optimizer.zero_grad()
                train_predictions = self.model(X_train_batch)
                train_loss = criterion(train_predictions, y_train_batch)
                train_loss.backward()
                optimizer.step()
                scheduler.step()

                # Calculate training loss
                epoch_loss += train_loss.item()

                # Validation step
                val_predictions = self.model(X_val_batch)
                val_rmse = self.calculate_rmse(val_predictions, y_val_batch).item()
                epoch_rmse += val_rmse  # Accumulate RMSE for this batch

            # Time cost for the epoch
            epoch_time = time.time() - start_time

            # Print epoch statistics
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/batch_num:.4f}, Validation RMSE: {epoch_rmse/batch_num:.4f}, Time: {epoch_time:.2f} seconds")

        return self.model

    def test(self):
        '''
        Tests the machine learning model on the testing data.

        Returns:
            numpy.ndarray: The predicted results on the testing data.
        '''
        pass

    def save_png(self, final_mapper_df, target_date, model_file_name):
        # Plotting the SWE results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            final_mapper_df['Longitude'], 
            final_mapper_df['Latitude'], 
            c=final_mapper_df['SWE_predicted'], 
            cmap='viridis', 
            s=10
        )
        plt.colorbar(scatter, label="SWE (mm)")
        plt.title(f"SWE {target_date} {model_file_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plot_path = f"{plot_dir}/swe_map_{target_date}_{model_file_name}.png"
        plt.savefig(plot_path)
        # plt.show()

        print(f"SWE map saved to {plot_path}")

    def predict_swe_map(
        self, target_date: str, model_path: str = None, 
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

            # Step 4: Extract the features
            X = batch[['Latitude', 'Longitude', 'day_of_year', 'month', 'year']].values
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


    def predict(
        self, 
        date, 
        csv_path: str = None, 
        save_csv: bool = False, 
        tiff: bool = False,
        skip_exists: bool = False,
    ):
        '''
        Makes predictions using the trained model on new input data.

        Args:
            input_x (numpy.ndarray): The input data for prediction.

        Returns:
            numpy.ndarray: The predicted results.
        '''
        tif_path = f"{output_dir}/{self.holename}_e{self.epochs}_n{self.normalization}_nn_swe_predicted_{date}.tif"
        if csv_path is None:
            csv_path = f"{output_dir}/{self.holename}_e{self.epochs}_n{self.normalization}_nn_swe_predicted_{date}.csv"
        
        if skip_exists:
            if os.path.exists(tif_path):
                print(f"Skip as {tif_path} exists")
                return
            if os.path.exists(csv_path):
                print(f"Skip as {csv_path} exists")
                return

        final_predicted_df = self.predict_swe_map(
            date, 
            model_path = self.model_path,
        )
        if save_csv:
            final_predicted_df.to_csv(csv_path, index=False)
            print("saved to csv: ", csv_path)

            if tiff:
                # Save geotiff
                csv_to_geotiff(
                    csv_file = csv_path, 
                    output_tiff=tif_path,
                )
        return final_predicted_df

    def post_processing(self):
        '''
        Perform post-processing on the model's predictions.

        Returns:
            None
        '''
        pass


if __name__ == "__main__":

    # Define start and end date for training
    start_date_str = '2024-10-01'
    end_date_str = '2025-01-15'
    
    hole_model = BaseNNHole(
        start_date_str = test_start_date,
        end_date_str = test_end_date,
        batch_size = 310000,
        epochs = 1,
        train_ratio = 0.01,
        test_ratio = 0.2,
        val_ratio = 0.2,
        retrain = False,
        normalization = False,
    )
    
    hole_model.preprocessing(verbose=True)
    hole_model.train()
    hole_model.save()
    hole_model.evaluate()
    hole_model.predict("2025-01-29", save_csv=True, tiff=True)



    

