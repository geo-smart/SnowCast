
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
from snowcast_utils import homedir, work_dir

import pandas as pd
import numpy as np
import xarray as xr
import sys
import time
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import os

matplotlib.use('Agg')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
SEQUENCE_LENGTH = 7
TARGET_COLUMN = 'swe_value'
ZERO_DROP_RATIO = 0.5

BASE_FEATURES = [
    'SWE', 
    #'change_in_swe_inch', 'snow_depth', 'air_temperature_observed_f',
    'precipitation_amount', 'relative_humidity_rmin', 'potential_evapotranspiration',
    'air_temperature_tmmx', 'relative_humidity_rmax', 'mean_vapor_pressure_deficit',
    'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect', 'Curvature',
    'Northness', 'Eastness', 'fsca', 'Slope'
]

# ET can reach 0.82 and 2.2
# selected_columns = [
#   'swe_value',
#   'SWE',
#   'fsca',
#   'air_temperature_tmmx', 
#   'air_temperature_tmmn', 
#   'potential_evapotranspiration', 
#   'relative_humidity_rmax', 
#   'Elevation',	
#   'Slope',	
#   'Curvature',	
#   'Aspect',	
#   'Eastness',	
#   'Northness',
# ]

# Derived features include lagged values
# FEATURES = BASE_FEATURES + [f"{feature}_lag{i}" for feature in BASE_FEATURES for i in range(1, SEQUENCE_LENGTH + 1)]
FEATURES = BASE_FEATURES

import torch
import subprocess

def get_available_cuda(min_free_mem_gb=10):
    """
    Returns the index of an available GPU with at least min_free_mem_gb free.
    Tries all GPUs in order.
    """
    if not torch.cuda.is_available():
        print("⚠️ No CUDA available, using CPU")
        return torch.device("cpu")
    
    try:
        # Query GPU memory using nvidia-smi
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        )
        free_mems = [int(x) for x in smi_output.decode("utf-8").strip().split("\n")]

        for i, free_mem in enumerate(free_mems):
            if free_mem >= min_free_mem_gb * 1024:  # convert GB → MB
                print(f"🔹 Using GPU {i} with {free_mem} MB free")
                return torch.device(f"cuda:{i}")

        print("⚠️ No GPU has enough free memory, using CPU")
        return torch.device("cpu")

    except Exception as e:
        print("⚠️ Could not query GPUs:", e)
        return torch.device("cpu")

device = get_available_cuda()

n_gpus = torch.cuda.device_count()
print(f"🔹 Found {n_gpus} GPUs")

# Data Preprocessing
def load_data(filepath):
    print("Loading dataset...")

    df = pd.read_csv(filepath)
    df = df[df[TARGET_COLUMN] >= 0]
    df = df.sort_values(by=["station_name", "date"])

    print(df.columns)
    df = df.fillna(-1)
    print("Assign -1 to fsca and SWE column where values are >100..")
    df.loc[df['fsca'] > 100, 'fsca'] = -1
    df.loc[df['SWE'] > 100, 'SWE'] = -1
    
    # df = df.iloc[:1000000]  # Limit rows for faster processing
    return df

def load_dataset_from_netcdf(nc_path):
    print(f"📦 Loading NetCDF dataset: {nc_path}")
    ds = xr.open_dataset(nc_path)
    print(f"✅ Dataset loaded with dimensions: {ds.dims}")
    print(f"Variables: {list(ds.data_vars)}")
    return ds


import pandas as pd
import dask.dataframe as dd
import xarray as xr
import numpy as np

def csv_to_netcdf(filepath, out_path, time_col="date"):
    print("🔄 Reading CSV...")
    df = load_data(filepath, index=0)

    print(df.describe())
    if "ndsi" in df.columns:
        print(df["ndsi"].describe())
        print("Unique values in ndsi:", df["ndsi"].nunique())
        print("Non-zero count:", (df["ndsi"] != 0).sum())

    # station_id = lat_lon
    print("🔄 Creating station IDs...")
    df["station_id"] = df["lat"].astype(str) + "_" + df["lon"].astype(str)

    # convert time
    df[time_col] = pd.to_datetime(df[time_col])

    # check duplicates
    dupes = df.duplicated(subset=[time_col, "station_id"], keep=False)
    if dupes.any():
        print(f"⚠️ Found {dupes.sum()} duplicate rows for (time, station_id). Keeping first occurrence.")
        df = df.drop_duplicates(subset=[time_col, "station_id"], keep="first")

    # full index (time × station_id)
    all_times = np.sort(df[time_col].unique())
    all_stations = df["station_id"].unique()

    dataset_vars = {}
    for col in df.columns:
        if col in [time_col, "lat", "lon", "station_name", "station_id"]:
            continue
        print(f"⏱️ Processing variable: {col}")

        pivot = df.pivot(index=time_col, columns="station_id", values=col)  # no aggregation
        pivot = pivot.reindex(index=all_times, columns=all_stations)
        # Replace NaN with -1
        pivot = pivot.fillna(-1)
        print(f"↪ pivot shape = {pivot.shape}, {pivot.notna().sum().sum()} non-NA values")

        dataset_vars[col] = (("time", "station_id"), pivot.values)

    print("📦 Building xarray Dataset...")
    ds = xr.Dataset(
        dataset_vars,
        coords={
            "time": all_times,
            "station_id": all_stations
        }
    )

    # print metadata stats
    for var in ds.data_vars:
        arr = ds[var].values
        nan_count = np.isnan(arr).sum()
        finite_arr = arr[np.isfinite(arr)]
        vmin = finite_arr.min() if finite_arr.size > 0 else None
        vmax = finite_arr.max() if finite_arr.size > 0 else None
        mean = finite_arr.mean() if finite_arr.size > 0 else None

        print(f"🔹 {var}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"   - NaN count: {nan_count:,}")
        print(f"   - Min: {vmin}")
        print(f"   - Max: {vmax}")
        print(f"   - Mean: {mean}")
        print()

    ds.to_netcdf(out_path)
    print(f"✅ Saved NetCDF: {out_path}")


import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

def array_stats(name, arr):
    total = arr.size
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    n_valid = total - n_nan - n_inf
    print(f"📊 {name} statistics:")
    print(f"   Shape: {arr.shape}")
    print(f"   Total elements: {total:,}")
    print(f"   NaN count: {n_nan:,} ({n_nan/total:.2%})")
    print(f"   Inf count: {n_inf:,} ({n_inf/total:.2%})")
    print(f"   Valid count: {n_valid:,} ({n_valid/total:.2%})")
    print()


class TimeSeriesNetCDFDataset(Dataset):

    def __init__(
        self, nc_path, features, target_feature, sequence_length, forecast_horizon=7, 
        device="cpu", chunk_size=1000, cache_path=None, profile=False
    ):
        """
        PyTorch Dataset for large NetCDF time series.
        Processes data in chunks to avoid memory issues.
        Only keeps sequences where the target forecast horizon contains finite, non-negative values.
        """
        self.ds = xr.open_dataset(nc_path)
        self.features = features
        self.target_feature = target_feature
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.chunk_size = chunk_size
        self.profile = profile  # ⏱️ profiling flag

        self.num_time = self.ds[features[0]].shape[0]
        self.num_station = self.ds[features[0]].shape[1]
        self.num_feature = len(features)

        print(f"📦 NetCDF loaded. Time steps: {self.num_time}, Stations: {self.num_station}, Features: {self.num_feature}")

        # Use cache if provided and exists
        if cache_path is not None and os.path.exists(cache_path):
            print(f"🔄 Loading cached valid sequences from {cache_path}")
            self.valid_sequences = np.load(cache_path, allow_pickle=True).tolist()
        else:
            print("🔹 Generating valid sequences...")
            self.valid_sequences = []

            for station_start in range(0, self.num_station, self.chunk_size):
                station_end = min(station_start + self.chunk_size, self.num_station)
                target_chunk = self.ds[target_feature].values[:, station_start:station_end]

                for station_offset in range(target_chunk.shape[1]):
                    station_idx = station_start + station_offset
                    for t in range(self.num_time - sequence_length - forecast_horizon + 1):
                        tgt_seq = target_chunk[t + sequence_length : t + sequence_length + forecast_horizon, station_offset]
                        if np.all(np.isfinite(tgt_seq)) and np.all(tgt_seq >= 0):
                            self.valid_sequences.append((station_idx, t))

                print(f"✅ Processed stations {station_start} - {station_end-1}, total valid sequences so far: {len(self.valid_sequences):,}")

            if cache_path is not None:
                np.save(cache_path, np.array(self.valid_sequences, dtype=object))
                print(f"💾 Cached valid sequences to {cache_path}")

        self.num_sequences = len(self.valid_sequences)
        print(f"🔹 Total valid sequences: {self.num_sequences}")

        # For profiling stats
        self.profile = profile
        if self.profile:
            self._time_costs = {"stack": [], "to_tensor": [], "getitem": []}

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if not self.profile:
            # --- normal path ---
            station_idx, time_start = self.valid_sequences[idx]
            seq = np.stack(
                [np.nan_to_num(
                    self.ds[f].values[time_start:time_start + self.sequence_length, station_idx],
                    nan=-1.0
                ) for f in self.features],
                axis=-1
            )
            tgt = np.nan_to_num(
                self.ds[self.target_feature].values[
                    time_start + self.sequence_length : time_start + self.sequence_length + self.forecast_horizon,
                    station_idx
                ],
                nan=-1.0
            )
            seq_tensor = torch.tensor(seq, dtype=torch.float32)   # stays on CPU
            tgt_tensor = torch.tensor(tgt, dtype=torch.float32)
            return seq_tensor, tgt_tensor

        # --- profiling path ---
        t0 = time.time()
        station_idx, time_start = self.valid_sequences[idx]

        # stack timing
        t1 = time.time()
        seq = np.stack(
            [np.nan_to_num(
                self.ds[f].values[time_start:time_start + self.sequence_length, station_idx],
                nan=-1.0
            ) for f in self.features],
            axis=-1
        )
        t2 = time.time()

        tgt = np.nan_to_num(
            self.ds[self.target_feature].values[
                time_start + self.sequence_length : time_start + self.sequence_length + self.forecast_horizon,
                station_idx
            ],
            nan=-1.0
        )

        # tensor conversion timing
        seq_tensor = torch.tensor(seq, dtype=torch.float32)   # stays on CPU
        tgt_tensor = torch.tensor(tgt, dtype=torch.float32)
        t3 = time.time()

        # record costs
        self._time_costs["stack"].append(t2 - t1)
        self._time_costs["to_tensor"].append(t3 - t2)
        self._time_costs["getitem"].append(t3 - t0)

        # log every 10000 samples
        if len(self._time_costs["getitem"]) % 10000 == 0:
            avg_get = np.mean(self._time_costs["getitem"][-10000:]) * 1000
            avg_stack = np.mean(self._time_costs["stack"][-10000:]) * 1000
            avg_tensor = np.mean(self._time_costs["to_tensor"][-10000:]) * 1000

            total_get = np.sum(self._time_costs["getitem"])  # cumulative cost
            print(
                f"⏱️ [Profiling @ {len(self._time_costs['getitem'])}] "
                f"GetItem(avg)={avg_get:.2f} ms | "
                f"Stack(avg)={avg_stack:.2f} ms | "
                f"ToTensor(avg)={avg_tensor:.2f} ms || "
                f"Total GetItem Time={total_get:.2f} s"
            )

        return seq_tensor, tgt_tensor

import torch
import torch.nn as nn

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, num_layers, output_dim, max_seq_len=500):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # Flexible positional encoding (initialize for max_seq_len)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.feature_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, _ = x.shape  # get actual sequence length
        # slice positional encoding to match T
        pe = self.positional_encoding[:, :T, :]
        x = self.input_embedding(x) + pe
        x = self.transformer_encoder(x)
        x = self.feature_pool(x.transpose(1, 2)).squeeze(-1)
        return self.output_layer(x)


def remove_nan(y, outputs):
    mask = torch.isfinite(y).all(dim=-1) & torch.isfinite(outputs).all(dim=-1)

    num_total = mask.numel()
    num_kept = mask.sum().item()
    num_filtered = num_total - num_kept

    if num_filtered > 0:
        print(f"filtered {num_filtered}/{num_total} rows")

    if mask.any():
        y = y[mask]
        outputs = outputs[mask]
    
    return y, outputs


import torch
import torch.nn as nn

import torch

def ndsi_elev_temp_to_swe_in_torch(ndsi, elev_m, temp_c,
                                   a=0.2, b=10.0, gamma=0.5, kappa=0.08, swe_max=200.0):
    """
    ndsi: tensor [B] or [B,H] values in [0,1]
    elev_m: tensor [B] (broadcastable to ndsi)
    temp_c: tensor [B] or [B,H] (broadcastable)
    returns: tensor of SWE in inches
    """
    ndsi_t = ndsi.float()
    elev_t = elev_m.float()
    temp_t = temp_c.float()

    baseline = float(a) + float(b) * ndsi_t
    elev_factor = 1.0 + float(gamma) * (elev_t / 1000.0)
    # broadcast elev_factor if needed
    if elev_factor.dim() == 1 and ndsi_t.dim() == 2:
        elev_factor = elev_factor.unsqueeze(1)

    temp_factor = 1.0 - float(kappa) * torch.clamp(temp_t, min=0.0)
    temp_factor = torch.clamp(temp_factor, min=0.0)

    swe = baseline * elev_factor * temp_factor
    swe = torch.clamp(swe, min=0.0, max=float(swe_max))
    return swe

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss (weak prior):
      - Base loss vs. targets
      - Non-negativity penalty across horizon
      - Sign consistency: SWE shouldn't increase from t0->t1
        if yesterday had no snowfall or temps > 0°C
      - Elevation consistency: SWE should generally increase with elevation
    """

    FEATURE_IDX = {
        "precip": 1,
        "air_temperature_tmmx": 4,
        "air_temperature_tmmn": 7,
        "Elevation": 9,  # 👈 adjust index to match your BASE_FEATURES
    }

    def __init__(self, base_loss, lambda_nonneg=0.01, lambda_sign=0.01, lambda_elev=0.01, snow_temp_c=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_nonneg = lambda_nonneg
        self.lambda_sign = lambda_sign
        self.lambda_elev = lambda_elev
        self.snow_temp_c = snow_temp_c

    def forward(self, inputs, outputs, targets):
        """
        inputs:  (B, past_seq, F)
        outputs: (B, H)   predicted SWE forecast horizon
        targets: (B, H)   ground truth SWE
        """
        device = outputs.device
        base_loss_val = self.base_loss(outputs, targets)

        # 1. Non-negativity penalty
        nonneg_loss = torch.mean(torch.relu(-outputs))

        # 2. Sign consistency (only day-1 forecast)
        delta_swe_first = outputs[:, 1] - outputs[:, 0]  # (B,)
        precip_yest = inputs[:, -1, self.FEATURE_IDX["precip"]].to(device)
        tmax_yest = inputs[:, -1, self.FEATURE_IDX["air_temperature_tmmx"]].to(device)
        tmin_yest = inputs[:, -1, self.FEATURE_IDX["air_temperature_tmmn"]].to(device)

        temp_yest = ((tmax_yest + tmin_yest) / 2.0 - 32.0) * (5.0 / 9.0)
        warm_dry_mask = ((precip_yest < 1e-3) & (temp_yest > self.snow_temp_c)).float()
        sign_loss = torch.mean(torch.relu(delta_swe_first * warm_dry_mask))

        # 3. Elevation consistency (weak monotonic prior)
        # Take last timestep’s elevation
        elev = inputs[:, -1, self.FEATURE_IDX["Elevation"]].to(device)  # (B,)
        # Normalize elevation to [0,1] to avoid scale explosion
        elev_norm = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)
        # Expected SWE: higher elevation → higher SWE
        # Penalize when SWE is low despite high elevation
        swe_first = outputs[:, 0]  # SWE forecast at first horizon
        elev_loss = torch.mean(torch.relu(elev_norm - swe_first / (swe_first.max() + 1e-6)))

        # Total
        total_loss = (
            base_loss_val
            + self.lambda_nonneg * nonneg_loss
            + self.lambda_sign * sign_loss
            + self.lambda_elev * elev_loss
        )
        return total_loss


def drop_invalid_rows(x, y, invalid_val=-1):
    """
    Remove rows where any feature equals invalid_val.
    Supports 2D [N, D] and 3D [N, T, D].
    """
    if x.dim() == 2:  # [N, D]
        mask = ~(x == invalid_val).any(dim=1)
    elif x.dim() == 3:  # [N, T, D]
        mask = ~((x == invalid_val).any(dim=2).any(dim=1))
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")

    return x[mask], y[mask]

# Training and Feature Extraction
def train_transformer_feature_extractor(
    model, cache_folder, criterion, optimizer, epochs, device, 
    accum_steps=30 # 👈 new param
):   
    """
    Trains the transformer model with the given data and returns the training
    and validation loss over epochs.
    """
    train_loss_list = []
    val_loss_list = []

    print(f"🔹 Using device: {device}")
    if n_gpus > 1:
        print(f"✅ Using {n_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)  # will now only use GPU1 & GPU2

    model = model.to(device)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{epochs}")
        print("-" * 60)

        # ==================== Training phase ==================== #
        model.train()
        train_loss = 0.0
        batch_count = 0
        epoch_start = time.time()

        accum_x, accum_y = [], []

        for batch_idx, (x, y) in enumerate(iterate_cached_batches(cache_folder, "train")):
            x, y = x.to(device).float(), y.to(device).float()
            x, y = drop_invalid_rows(x, y, invalid_val=-1)
            if x.numel() == 0:
                continue

            # Collect micro-batches
            accum_x.append(x)
            accum_y.append(y)

            if len(accum_x) == accum_steps:
                big_x = torch.cat(accum_x, dim=0)
                big_y = torch.cat(accum_y, dim=0)
                accum_x, accum_y = [], []

                optimizer.zero_grad()
                outputs = model(big_x)
                big_y, outputs = remove_nan(big_y, outputs)
                loss = criterion(outputs, big_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_count += 1

                if batch_count % 50 == 0:
                    print(f"[Batch {batch_idx:5d}] Loss={loss.item():.4f}")

        # Flush leftover micro-batches at end of epoch
        if accum_x:
            big_x = torch.cat(accum_x, dim=0)
            big_y = torch.cat(accum_y, dim=0)
            optimizer.zero_grad()
            outputs = model(big_x)
            big_y, outputs = remove_nan(big_y, outputs)
            loss = criterion(outputs, big_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / max(1, batch_count)
        train_loss_list.append(avg_train_loss)

        # ==================== Validation phase ==================== #
        model.eval()
        val_loss, batch_count = 0.0, 0

        with torch.no_grad():
            for x, y in iterate_cached_batches(cache_folder, "val"):
                x, y = x.to(device).float(), y.to(device).float()
                x, y = drop_invalid_rows(x, y, invalid_val=-1)
                if x.numel() == 0:
                    continue

                outputs = model(x)
                y, outputs = remove_nan(y, outputs)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                batch_count += 1

        avg_val_loss = val_loss / max(1, batch_count)
        val_loss_list.append(avg_val_loss)

        # ==================== End of Epoch Log ==================== #
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  🟢 Training Loss:   {avg_train_loss:.6f}")
        print(f"  🔵 Validation Loss: {avg_val_loss:.6f}")
        print(f"  ⏱️ Epoch Time: {time.time()-epoch_start:.2f}s")

    return train_loss_list, val_loss_list

def extract_features(model, cache_folder, prefix="train"):
    model.eval()
    features_list, targets_list = [], []
    with torch.no_grad():
        # for x, y in data_loader:
        for batch_idx, (x, y) in enumerate(iterate_cached_batches(cache_folder, prefix)):
            x, y = x.to(device), y.to(device)
            
            x, y = drop_zeros(x, y, ZERO_DROP_RATIO)

            # print("input shape: ", x.shape, " - target shape: ", y.shape)
            # Extract features from the model
            features = model(x)  # Shape: [batch_size, seq_length]

            # Reshape features to [batch_size, seq_length, 1]
            features_reshaped = features.unsqueeze(2)  # Add a new dimension, resulting in [batch_size, seq_length, 1]

            # Merge features with x along the feature dimension
            combined_features = torch.cat((x, features_reshaped), dim=2)  # Shape: [batch_size, seq_length, feature_dim + 1]

            # Add combined features to the list
            features_list.append(combined_features.cpu().numpy())

            # Use only the first day of the target list
            targets_list.append(y[:, 0].cpu().numpy())

    # Concatenate all features and targets into single arrays
    concatenated_features = np.concatenate(features_list, axis=0)
    concatenated_targets = np.concatenate(targets_list, axis=0)

    # Flatten the last two dimensions of concatenated_features
    flattened_features = concatenated_features.reshape(concatenated_features.shape[0], -1)

    # Print shapes for debugging
    print(f"Original concatenated features shape: {concatenated_features.shape}")
    print(f"Flattened features shape: {flattened_features.shape}")
    print(f"Extracted targets shape: {concatenated_targets.shape}")

    return flattened_features, concatenated_targets

# def extract_features(model, data_loader):
#     model.eval()
#     features_list, targets_list = [], []
#     with torch.no_grad():
#         for x, y in data_loader:
#             features = model(x)
#             features_list.append(features.numpy())
#             targets_list.append(y.numpy())
#     return np.concatenate(features_list, axis=0), np.concatenate(targets_list, axis=0)

def plot_comparison_chart(test_targets, test_preds, plot_dir):
    """
    Plots a comparison chart of predicted vs actual values and saves it to the plot directory.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets.flatten(), test_preds.flatten(), alpha=0.6, label="Predicted vs Actual")
    plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 
             color='red', linestyle='--', label="Ideal Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Comparison of Predicted vs Actual Values")
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_plot_path = f"{plot_dir}/novel_comparison_plot_{timestamp}.png"
    plt.savefig(comparison_plot_path)
    print(f"Comparison chart saved at {comparison_plot_path}")
    # plt.show()
    plt.close()

def plot_learning_curve(train_loss, val_loss, plot_dir, title=""):
    """
    Plots the learning curve (training and validation loss over epochs) and saves it to the plot directory.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve {title}")
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    learning_curve_path = f"{plot_dir}/novel_learning_curve_{timestamp}.png"
    plt.savefig(learning_curve_path)
    print(f"Learning curve saved at {learning_curve_path}")
    # plt.show()
    plt.close()

def collate_fn(batch):
    # Remove None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # or raise an error
    return torch.utils.data.dataloader.default_collate(batch)

def preprocess_training_transformer_tensor_files(nc_filepath, out_dir,):
    dataset = TimeSeriesNetCDFDataset(
        nc_filepath, FEATURES, TARGET_COLUMN, SEQUENCE_LENGTH, forecast_horizon=7, 
        device=device, cache_path=f"{nc_filepath}.npy", profile=True
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    batch_size = 50000
    print("Batch size: ", batch_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        # num_workers=4,          # number of worker processes
        # pin_memory=True,        # speeds up host→GPU transfers
        # prefetch_factor=2       # each worker preloads batches
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        # num_workers=4,          # number of worker processes
        # pin_memory=True,        # speeds up host→GPU transfers
        # prefetch_factor=2       # each worker preloads batches
    )
    # print(f"Train size: {train_size}, Validation size: {val_size}")
    cache_raw_batches(train_loader, out_dir, prefix="train")
    cache_raw_batches(val_loader, out_dir, prefix="val")

def cache_raw_batches(dataloader, out_dir, prefix="train", device="cpu"):
    os.makedirs(out_dir, exist_ok=True)

    for i, (x, y) in enumerate(dataloader):
        cache_file = os.path.join(out_dir, f"{prefix}_batch{i:06d}.pt")

        # Skip if cache already exists
        if os.path.exists(cache_file):
            continue

        # Move to CPU before saving (saves GPU memory and ensures portability)
        x, y = x.to(device if device == "cpu" else "cpu"), y.to("cpu")

        torch.save((x, y), cache_file)

        if i % 50 == 0:
            print(f"💾 Saved {cache_file}")

    print(f"✅ Finished caching {prefix} with {i+1} batches")

import glob
import torch

# def iterate_cached_batches(cache_dir, prefix="train"):
#     """Generator that yields (x, y) tensors from cached batch files."""
#     batch_files = sorted(glob.glob(f"{cache_dir}/{prefix}_batch*.pt"))
#     for fname in batch_files:
#         x, y = torch.load(fname)
#         yield x, y


# def iterate_cached_batches(cache_dir, prefix="train", strategy="mean"):
#     """
#     Generator that yields (x, y) tensors from cached batch files,
#     with missing values (-1) imputed.
#     """
#     batch_files = sorted(glob.glob(f"{cache_dir}/{prefix}_batch*.pt"))
#     for fname in batch_files:
#         x, y = torch.load(fname)

#         # Imputation
#         # mask = (x != -1)
#         # x_imp = x.clone().float()

#         # if strategy == "mean":
#         #     # feature-wise mean imputation across the batch
#         #     for f in range(x.shape[1]):
#         #         valid_vals = x_imp[:, f][mask[:, f]]
#         #         if valid_vals.numel() > 0:
#         #             fill_val = valid_vals.mean()
#         #             x_imp[:, f][~mask[:, f]] = fill_val
#         # elif strategy == "zero":
#         #     x_imp[~mask] = 0.0
#         # elif strategy == "median":
#         #     for f in range(x.shape[1]):
#         #         valid_vals = x_imp[:, f][mask[:, f]]
#         #         if valid_vals.numel() > 0:
#         #             fill_val = valid_vals.median()
#         #             x_imp[:, f][~mask[:, f]] = fill_val
#         # else:
#         #     raise ValueError(f"Unknown imputation strategy: {strategy}")

#         yield x, y

import torch
import glob

def drop_zeros(x, y, drop_ratio=ZERO_DROP_RATIO):
    # --- Drop 50% of rows with y == 0 ---
    zero_mask = (y == 0)
    zero_indices = torch.nonzero(zero_mask, as_tuple=False).squeeze()

    if zero_indices.numel() > 0:
        # randomly select ~50% of zero rows to drop
        n_drop = int(len(zero_indices) * drop_ratio)
        drop_idx = zero_indices[torch.randperm(len(zero_indices))[:n_drop]]

        keep_mask = torch.ones(len(y), dtype=torch.bool)
        keep_mask[drop_idx] = False

        x = x[keep_mask]
        y = y[keep_mask]
    
    return x, y

def iterate_cached_batches(cache_dir, prefix="train", strategy="mean", drop_ratio=ZERO_DROP_RATIO):
    """
    Generator that yields (x, y) tensors from cached batch files,
    with missing values (-1) imputed, and optionally dropping a portion
    of rows where y == 0.
    """
    batch_files = sorted(glob.glob(f"{cache_dir}/{prefix}_batch*.pt"))
    for fname in batch_files:
        x, y = torch.load(fname)  # shapes: (B, F, ...) for x, (B,) or (B, H) for y

        x, y = drop_zeros(x, y, drop_ratio)

        yield x, y



# Main Workflow
def do_ensemble_model(cache_folder, plot_dir, 
    extractor_train_toggle=True, 
    extractor_model_path=None,
    visionary_train_toggle=True,
    visionary_model_path=None,
):
    """
    Performs ensemble modeling and visualizes results including:
    - R2 score
    - Comparison plots
    - Learning curves
    """

    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size

    print("Step 3: Initializing the transformer model...")
    transformer = TransformerFeatureExtractor(
        input_dim=len(FEATURES), nhead=4, hidden_dim=64, num_layers=2, output_dim=7
    )
    
    print("Step 4: Training the transformer model...")
    if extractor_train_toggle:
        epochs = 200
        # criterion = nn.MSELoss()
        # criterion = PhysicsInformedLoss(nn.MSELoss(), lambda_nonneg=0.1, lambda_balance=0.05)
        criterion = PhysicsInformedLoss(
            base_loss=nn.MSELoss(),
            # lambda_nonneg=0.1,
            # lambda_balance=0.05,
            # melt_factor=0.004  # tweakable
        )

        optimizer = optim.Adam(transformer.parameters(), lr=1e-3)
        print("Transformer model, loss function, and optimizer initialized.")

        train_loss, val_loss = train_transformer_feature_extractor(
            transformer, cache_folder, 
            criterion, optimizer, 
            epochs=epochs, device=device
        )
        print("Transformer training completed.")
    else:
        transformer.load_state_dict(torch.load(extractor_model_path, map_location=device))
        transformer.to(device)
        print("Transformer training loaded from ", extractor_model_path)

    print("Step 5: Extracting features from the transformer...")
    train_features, train_targets = extract_features(transformer, cache_folder, prefix="train")
    test_features, test_targets = extract_features(transformer, cache_folder, prefix="val")
    print(f"Feature extraction complete. Train shape: {train_features.shape}, Test shape: {test_features.shape}")

    print("Step 6: Training the XGBoost model on extracted features...")
    print(f"Training data shape for XGBoost - Features: {train_features.shape}, Targets: {train_targets.shape}")
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                                 n_estimators=100, 
                                 max_depth=6, 
                                 learning_rate=0.1)
    if visionary_train_toggle:
        xgb_model.fit(train_features, train_targets)
        print("XGBoost training completed.")
    else:
        xgb_model.load_model(visionary_model_path)
        print("XGBoost training loaded from ", visionary_model_path)

    # Step 7: Saving the models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(plot_dir, "../models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save the transformer model
    if extractor_train_toggle:
        transformer_save_path = os.path.join(model_save_dir, f"ensemble_transformer_model_{timestamp}.pth")
        torch.save(transformer.state_dict(), transformer_save_path)
        print(f"Transformer model saved at: {transformer_save_path}")
    
    # Save the XGBoost model
    if visionary_train_toggle:
        xgb_model_save_path = os.path.join(model_save_dir, f"ensemble_xgboost_model_{timestamp}.json")
        xgb_model.save_model(xgb_model_save_path)
        print(f"XGBoost model saved at: {xgb_model_save_path}")

    print("Step 8: Evaluating the XGBoost model...")
    print(f"Test data shape for XGBoost prediction - Features: {test_features.shape}, Targets: {test_targets.shape}")
    test_preds = xgb_model.predict(test_features)

    print_array_stats("Test Targets", test_targets)
    print_array_stats("Test Predictions", test_preds)

    print(f"Shape of predictions: {test_preds.shape}")

    test_features_reshaped = test_features.reshape(test_features.shape[0], SEQUENCE_LENGTH, 17)
    transformer_predictions = test_features_reshaped[:, 0, -1]

    rmse = np.sqrt(mean_squared_error(test_targets, transformer_predictions))
    r2 = r2_score(test_targets, transformer_predictions)
    print(f"Transformer RMSE: {rmse:.4f} R2 Score: {r2:.4f}")

    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    r2 = r2_score(test_targets, test_preds)
    print(f"XGBoost RMSE: {rmse:.4f} R2 Score: {r2:.4f}")


    print("Step 9: Generating plots...")
    plot_comparison_chart(test_targets, test_preds, plot_dir)
    if extractor_train_toggle:
        plot_learning_curve(train_loss, val_loss, plot_dir, title=f" transformer + xgboost physics added {test_features_reshaped.shape}")
    print(f"Plots saved in: {plot_dir}")


def print_array_stats(name, arr):
    print(f"\n📊 Statistics for {name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min:   {np.min(arr):.6f}")
    print(f"  Max:   {np.max(arr):.6f}")
    print(f"  Mean:  {np.mean(arr):.6f}")
    print(f"  Std:   {np.std(arr):.6f}")

# Main Script
if __name__ == "__main__":
    # filepath = f'{homedir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
    # filepath = f'{homedir}/data/gridmet_test_run/final_merged_data_4yrs_snotel_and_ghcnd_stations_2002-01-01_2024-12-31.csv_sorted_slope_corrected.csv_4yrs_time_series.csv_gap_filled.csv'

    csv_file_path = f'{homedir}/data/gridmet_test_run/all_points_final_merged_training_2002-01-01_2024-12-31_snodas_mask.csv'
    nc_filepath = f'{homedir}/data/gridmet_test_run/all_points_final_merged_training_2002-01-01_2024-12-31_snodas_mask.nc'

    # csv_file_path = f"{homedir}/data/gridmet_test_run/train_bttf_1mln.csv"
    # nc_filepath = f"{homedir}/data/gridmet_test_run/train_bttf_1mln.csv_preprocessed6.nc"
    # nc_filepath = f"{homedir}/data/gridmet_test_run/final_merged_data_4yrs_snotel_and_ghcnd_stations_2002-01-01_2024-12-31.csv_sorted_slope_corrected.csv_4yrs_time_series.csv_gap_filled.csv_preprocessed_50_mln.nc"
    # nc_filepath = f"{homedir}/data/gridmet_test_run/train_bttf_1mln.nc"
    plot_dir = f'{homedir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    # cache_out_dir = f"{homedir}/data/ml_tensor_cache/"
    cache_out_dir = f"{homedir}/data/ml_tensor_cache_50mln_snodas/"
    os.makedirs(cache_out_dir, exist_ok=True)

    csv_to_netcdf(csv_file_path, out_path=nc_filepath, time_col="date")
    # preprocess_training_transformer_tensor_files(nc_filepath, cache_out_dir, )
    # do_ensemble_model(
    #     cache_out_dir, plot_dir,
    #     extractor_train_toggle=True,
    #     extractor_model_path=None,
    #     # extractor_model_path="/home/chetana/plots/../models/ensemble_transformer_model_20250906_025046.pth", # epoch 200 model
    #     # extractor_model_path="/home/chetana/models/ensemble_transformer_model_20250914_035616.pth",

    #     visionary_train_toggle=True,
    #     # visionary_model_path="/home/chetana/plots/../models/ensemble_xgboost_model_20250914_035616.json",
    #     visionary_model_path=None,
    # )

