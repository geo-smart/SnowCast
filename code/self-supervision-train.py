import time
import torch
import torch.nn as nn


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

from bttf_train import BASE_FEATURES, TransformerFeatureExtractor, FEATURES, PhysicsInformedLoss, get_available_cuda, iterate_cached_batches, SEQUENCE_LENGTH, remove_nan, extract_features, plot_comparison_chart, plot_learning_curve, print_array_stats

MASK_PROB = 0.15  # percent of features to mask for self-supervision

# device = get_available_cuda()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpus = torch.cuda.device_count()
# print(f"🔹 Found {n_gpus} GPUs")
# Restrict to GPU 1 and GPU 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"🔹 Found {n_gpus} GPUs")

# ================================
# 🔹 Updated Transformer with Dual Heads
# ================================
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔹 Two heads:
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)   # self-supervised
        self.output_head = nn.Linear(hidden_dim, output_dim)          # supervised

        self.feature_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, self_supervised=False):
        x = self.input_embedding(x) + self.positional_encoding
        z = self.transformer_encoder(x)

        if self_supervised:
            # return per-token reconstruction [batch, seq_len, input_dim]
            return self.reconstruction_head(z)
        else:
            # pooled features → supervised prediction
            pooled = self.feature_pool(z.transpose(1, 2)).squeeze(-1)
            return self.output_head(pooled)

def mask_inputs(x, mask_prob=MASK_PROB):
    """
    Randomly masks features in the input tensor.
    x: [batch, seq_len, features]
    Returns (masked_x, mask)
    """
    mask = torch.rand_like(x) < mask_prob
    masked_x = x.clone()
    masked_x[mask] = 0.0  # could also replace with noise or learnable [MASK] token
    return masked_x, mask

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


# ================================
# 🔹 Training Loop (unchanged except pred call)
# ================================
def train_transformer_feature_extractor(
    model, cache_folder, optimizer, epochs, device, 
    self_supervised=True, accum_steps=4   # 👈 new param
):
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
                if self_supervised:
                    masked_x, mask = mask_inputs(big_x)
                    masked_x, mask = masked_x.to(device), mask.to(device)
                    pred = model(masked_x, self_supervised=True)
                    loss = criterion(pred[mask], big_x[mask])
                else:
                    outputs = model(big_x, self_supervised=False)
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
            if self_supervised:
                masked_x, mask = mask_inputs(big_x)
                masked_x, mask = masked_x.to(device), mask.to(device)
                pred = model(masked_x, self_supervised=True)
                loss = criterion(pred[mask], big_x[mask])
            else:
                outputs = model(big_x, self_supervised=False)
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

                if self_supervised:
                    masked_x, mask = mask_inputs(x)
                    masked_x, mask = masked_x.to(device), mask.to(device)
                    pred = model(masked_x, self_supervised=True)
                    loss = criterion(pred[mask], x[mask])
                else:
                    outputs = model(x, self_supervised=False)
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


# Main Workflow
def do_ensemble_model(cache_folder, plot_dir, 
    extractor_train_toggle=True, 
    extractor_model_path=None,
    visionary_train_toggle=True, 
    visionary_model_path=None,
):
    """
    Performs ensemble modeling with:
    - Transformer pretraining (self-supervised)
    - Transformer fine-tuning (supervised)
    - XGBoost on extracted transformer features
    - Evaluation and visualization
    """

    print("Step 3: Initializing the transformer model...")
    transformer = TransformerFeatureExtractor(
        input_dim=len(FEATURES), nhead=4, hidden_dim=64, num_layers=2, output_dim=7
    )

    if extractor_train_toggle:
        # ===========================
        # 1) Self-Supervised Pretrain
        # ===========================
        print("Step 4a: Pretraining transformer (self-supervised)...")
        pretrain_epochs = 50
        optimizer = optim.Adam(transformer.parameters(), lr=1e-3)

        pre_train_loss, pre_val_loss = train_transformer_feature_extractor(
            transformer, cache_folder, optimizer, 
            epochs=pretrain_epochs, device=device, 
            self_supervised=True
        )
        print("✅ Transformer pretraining completed.")

        # ======================
        # 2) Supervised Fine-Tune
        # ======================
        print("Step 4b: Fine-tuning transformer (supervised)...")
        finetune_epochs = 10
        optimizer = optim.Adam(transformer.parameters(), lr=1e-4)  # lower LR

        ft_train_loss, ft_val_loss = train_transformer_feature_extractor(
            transformer, cache_folder, optimizer, 
            epochs=finetune_epochs, device=device, 
            self_supervised=False
        )
        print("✅ Transformer fine-tuning completed.")

        # merge learning curves for plotting
        train_loss = pre_train_loss + ft_train_loss
        val_loss   = pre_val_loss + ft_val_loss

    else:
        transformer.load_state_dict(torch.load(extractor_model_path, map_location=device))
        transformer.to(device)
        print("Transformer loaded from", extractor_model_path)

    # ==================================
    # 3) Feature Extraction for XGBoost
    # ==================================
    print("Step 5: Extracting features from the transformer...")
    train_features, train_targets = extract_features(transformer, cache_folder, prefix="train")
    test_features, test_targets   = extract_features(transformer, cache_folder, prefix="val")
    print(f"Feature extraction complete. Train shape: {train_features.shape}, Test shape: {test_features.shape}")

    # ===================
    # 4) XGBoost Training
    # ===================
    print("Step 6: Training the XGBoost model...")
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                                 n_estimators=100, 
                                 max_depth=6, 
                                 learning_rate=0.1)
    if visionary_train_toggle:
        xgb_model.fit(train_features, train_targets)
        print("XGBoost training completed.")
    else:
        xgb_model.load_model(visionary_model_path)
        print("XGBoost loaded from", visionary_model_path)

    # =================
    # 5) Save the models
    # =================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(plot_dir, "../models")
    os.makedirs(model_save_dir, exist_ok=True)

    if extractor_train_toggle:
        transformer_save_path = os.path.join(model_save_dir, f"ensemble_transformer_model_{timestamp}.pth")
        torch.save(transformer.state_dict(), transformer_save_path)
        print(f"Transformer model saved at: {transformer_save_path}")

    if visionary_train_toggle:
        xgb_model_save_path = os.path.join(model_save_dir, f"ensemble_xgboost_model_{timestamp}.json")
        xgb_model.save_model(xgb_model_save_path)
        print(f"XGBoost model saved at: {xgb_model_save_path}")

    # =====================
    # 6) Evaluation & Plots
    # =====================
    print("Step 8: Evaluating the ensemble...")
    test_preds = xgb_model.predict(test_features)

    print_array_stats("Test Targets", test_targets)
    print_array_stats("Test Predictions", test_preds)

    # transformer-only baseline (optional sanity check)
    test_features_reshaped = test_features.reshape(-1, 7, 17)
    transformer_predictions = test_features_reshaped[:, 0, -1]

    rmse = np.sqrt(mean_squared_error(test_targets, transformer_predictions))
    r2 = r2_score(test_targets, transformer_predictions)
    print(f"Transformer RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    r2 = r2_score(test_targets, test_preds)
    print(f"XGBoost RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

    print("Step 9: Generating plots...")
    plot_comparison_chart(test_targets, test_preds, plot_dir)

    if extractor_train_toggle:
        plot_learning_curve(
            train_loss, val_loss, plot_dir, 
            title=f"Transformer Pretrain+FineTune + XGBoost {test_features_reshaped.shape}"
        )

    print(f"✅ All plots saved in: {plot_dir}")


# Main Script
if __name__ == "__main__":
    # filepath = f'{homedir}/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
    # filepath = f'{homedir}/data/gridmet_test_run/final_merged_data_4yrs_snotel_and_ghcnd_stations_2002-01-01_2024-12-31.csv_sorted_slope_corrected.csv_4yrs_time_series.csv_gap_filled.csv'

    csv_file_path = f"{homedir}/data/gridmet_test_run/train_bttf_1mln.csv"
    # nc_filepath = f"{homedir}/data/gridmet_test_run/train_bttf_1mln.csv_preprocessed6.nc"
    # nc_filepath = f"{homedir}/data/gridmet_test_run/final_merged_data_4yrs_snotel_and_ghcnd_stations_2002-01-01_2024-12-31.csv_sorted_slope_corrected.csv_4yrs_time_series.csv_gap_filled.csv_preprocessed_50_mln.nc"
    # nc_filepath = f"{homedir}/data/gridmet_test_run/train_bttf_1mln.nc"
    plot_dir = f'{homedir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    # cache_out_dir = f"{homedir}/data/ml_tensor_cache/"
    cache_out_dir = f"{homedir}/data/ml_tensor_cache_50mln/"

    # csv_to_netcdf(csv_file_path, out_path=nc_filepath, time_col="date")
    # preprocess_training_transformer_tensor_files(nc_filepath, cache_out_dir, )
    do_ensemble_model(
        cache_out_dir, plot_dir,
        extractor_train_toggle=True,
        # extractor_model_path="/home/chetana/plots/../models/ensemble_transformer_model_20250906_025046.pth", # epoch 200 model
        extractor_model_path=None,
        visionary_train_toggle=True,
        visionary_model_path=None,
    )
