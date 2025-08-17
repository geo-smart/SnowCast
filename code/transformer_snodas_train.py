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
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datetime import datetime
import os
import pandas as pd
import numpy as np
import torch

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class SNODAS_Transformer_Model(nn.Module):
    def __init__(self, input_dim=8, embed_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, embed_dim)  # Project 8 features → embed_dim
        self.positional_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # input/output shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1)  # Predict scalar SWE
        )

        self.apply(self.init_weights)

    def forward(self, x):
        # x: (batch_size, seq_len=7, input_dim=8)
        x = self.input_layer(x)               # (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)      # add positional encoding

        x = self.transformer_encoder(x)      # (batch_size, seq_len, embed_dim)

        x = x.mean(dim=1)                    # global average pooling over seq_len -> (batch_size, embed_dim)

        x = self.output_layer(x)              # (batch_size, 1)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)

    def load_model(self, model_path):
        """Loads the model weights from a file."""
        try:
            state_dict = torch.load(model_path)
            self.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")


class SNODAS_Transformer_Hole(BaseNNHole):

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
        # random.shuffle(files)
        files = self.sort_files(files)

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
        # Normalize Year (assuming range ~[2000, 2100])
        df['year_enc'] = (df['year'] - 2000) / 100  

        # Normalize Month (1-12 → scaled between 0 and 1)
        df['month_enc'] = (df['month'] - 1) / 11  

        # Normalize Day of Year (1-365 → scaled between 0 and 1)
        df['day_enc'] = (df['day_of_year'] - 1) / 364  

        # Select final 8D input features
        return df[['Latitude', 'Longitude', 'year', 'month', 'day_of_year', 'year_enc', 'month_enc', 'day_enc']].values

    def extract_date(self, file_path):
        filename = os.path.basename(file_path)
        date_str = filename.split('_')[0]  # e.g. '2023-06-28'
        return datetime.strptime(date_str, "%Y-%m-%d")

    def sort_files(self, files):
        # Sort files by date extracted from filename
        files = sorted(files, key=self.extract_date)
        return files

    def load_batch_iterator(self, all_files, sequence_length=7, batch_size=8):
        """
        A generator that yields batches of (X_seq, y_seq) with:
        - X_seq: [batch_size * N, sequence_length, F]
        - y_seq: [batch_size * N, 1]
        
        Only loads (batch_size + sequence_length - 1) files per batch.
        """
        import pandas as pd
        import numpy as np
        import torch
        from datetime import datetime, timedelta
        import os

        assert sequence_length >= 1, "sequence_length must be ≥ 1"

        def extract_date(file_path):
            fname = os.path.basename(file_path)
            date_str = fname.split('_')[0]  # 'YYYY-MM-DD'
            return datetime.strptime(date_str, "%Y-%m-%d")

        # Sort input files by date
        all_files = sorted(all_files, key=extract_date)
        all_dates = [extract_date(f) for f in all_files]
        file_lookup = {extract_date(f): f for f in all_files}

        total_steps = len(all_files) - sequence_length

        for i in range(0, total_steps, batch_size):
            batch_X = []
            batch_y = []

            batch_dates = all_dates[i + sequence_length: i + sequence_length + batch_size]
            if len(batch_dates) < batch_size:
                break  # incomplete final batch

            for target_date in batch_dates:
                history_dates = [target_date - timedelta(days=d) for d in reversed(range(sequence_length))]
                if not all(d in file_lookup for d in history_dates + [target_date]):
                    print(f"[SKIP] Missing files for {target_date.date()} sequence")
                    continue

                # Load sequence
                df_seq = []
                for d in history_dates:
                    df = pd.read_csv(file_lookup[d])
                    df['date'] = pd.to_datetime(df['date'])
                    df['day_of_year'] = df['date'].dt.dayofyear
                    df['month'] = df['date'].dt.month
                    df['year'] = df['date'].dt.year
                    df_seq.append(df)

                encoded_seq = [
                    torch.tensor(self.encode_features(df) * 100, dtype=torch.float32)
                    for df in df_seq
                ]

                X = torch.stack(encoded_seq)  # [T, N, F]
                X = X.permute(1, 0, 2)        # [N, T, F]

                # Load target
                df_y = pd.read_csv(file_lookup[target_date])
                y_raw = df_y['snodas'].values
                y = np.log10(y_raw + 1)
                y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

                batch_X.append(X)
                batch_y.append(y)

            if not batch_X:
                continue

            # Concatenate over all N's in batch
            X_seq = torch.cat(batch_X, dim=0).to(self.device)  # [batch_size * N, T, F]
            y_seq = torch.cat(batch_y, dim=0).to(self.device)  # [batch_size * N, 1]

            print(f"[YIELD] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
            yield X_seq, y_seq



    # def load_batch(self, files):
    #     batch_data = []
        
    #     for file in files:
    #         if not os.path.exists(file):
    #             continue
    #         df = pd.read_csv(file)
    #         df['date'] = pd.to_datetime(df['date'])
    #         batch_data.append(df)
        
    #     batch_df = pd.concat(batch_data, ignore_index=True)

    #     # Extract date components
    #     batch_df['day_of_year'] = batch_df['date'].dt.dayofyear
    #     batch_df['month'] = batch_df['date'].dt.month
    #     batch_df['year'] = batch_df['date'].dt.year

    #     # Transform into 8D input features
    #     X = self.encode_features(batch_df)
    #     X *= 100

    #     # Log-transform the target variable
    #     y = np.log10(batch_df['snodas'] + 1)  # +1 to avoid log(0)
        
    #     return X, y

    def predict_swe_map(
        self, 
        target_date: str, 
        model_path: str = None, 
        batch_size: int = -1
    ):
        from tqdm import tqdm
        print("======>\nstart to predict SWE map...")

        if model_path is None:
            model_path = self.model_path
        self.model = self.load_model(model_path)
        self.model.eval()

        if batch_size == -1:
            batch_size = self.batch_size

        model_file_name = os.path.basename(model_path)
        print(f"Model loaded successfully. Running evaluation on {self.device}...")

        # Prepare files up to the target_date
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        all_files = [f for f in self.input_files if f"{target_date}" not in f and datetime.strptime(os.path.basename(f).split('_')[0], "%Y-%m-%d") <= date_obj]

        print(f"Total files for evaluation: {len(all_files)} (up to {target_date})")

        # Use batch iterator
        result_batches = []
        for X_batch, _ in tqdm(self.load_batch_iterator(all_files, sequence_length=7, batch_size=batch_size), desc="Running prediction"):
            with torch.no_grad():
                y_pred = self.model(X_batch).detach().cpu().numpy()

            y_pred = (10 ** y_pred) - 1  # Reverse log10 scaling
            result_batches.append(y_pred)

        # Flatten result and map back to lat/lon if needed
        all_preds = np.concatenate(result_batches, axis=0)  # [Total_points, 1]

        # Map predictions to location
        mapper_df = pd.read_csv(self.LAT_LON_CSV_PATH)
        if len(mapper_df) != len(all_preds):
            raise ValueError(f"Prediction size {len(all_preds)} doesn't match location data {len(mapper_df)}")

        mapper_df['SWE_predicted'] = all_preds.flatten()
        mapper_df['date'] = target_date

        if mapper_df.isna().any().any():
            raise ValueError("NaN detected in prediction output")

        self.save_png(mapper_df, target_date, model_file_name)
        return mapper_df




    def get_model(self):
        model = SNODAS_Transformer_Model().to(self.device)
        # Calculate the total number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")
        return model

    def train(self):
        print("======>\nstart to train...")
        model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        MAX_ROWS = 50000
        
        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}/{self.epochs}]")

            model.train()
            total_loss = 0.0
            total_samples = 0

            batch_iter = self.load_batch_iterator(self.train_files, sequence_length=7, batch_size=8)

            for i, (X_train, y_train) in enumerate(batch_iter):
                if X_train is None or y_train is None:
                    continue

                num_rows = X_train.shape[0]
                num_slices = (num_rows + MAX_ROWS - 1) // MAX_ROWS

                for j in range(num_slices):
                    start = j * MAX_ROWS
                    end = min(start + MAX_ROWS, num_rows)

                    X_batch = X_train[start:end].to(self.device)
                    y_batch = y_train[start:end].to(self.device)

                    print("X_batch.shape: ", X_batch.shape)

                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = loss_fn(pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    batch_size = X_batch.shape[0]
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                    # print(f"  └── Batch {i+1}.{j+1}: Loss = {loss.item():.4f} | Samples = {batch_size}")

            if total_samples > 0:
                avg_loss = total_loss / total_samples
                print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")
            else:
                print(f"[Epoch {epoch+1}] No valid samples in this epoch.")

    def new_batch_generator(self, files, batch_size, sequence_length=7):
        """
        Batch generator for Transformer input:
        - X: shape [batch_size, sequence_length, C, H, W]
        - y: shape [batch_size, 1] or [batch_size, 1, H, W]
        """

        assert sequence_length >= 1, "sequence_length must be at least 1"

        print("🚀 Starting batch generator with spatial input...")
        print(f"📁 Total files: {len(files)}")
        print(f"🧮 Sequence length: {sequence_length}, Batch size: {batch_size}")

        # Sort files by date
        def extract_date(f): return datetime.strptime(os.path.basename(f).split("_")[0], "%Y-%m-%d")
        files = sorted(files, key=extract_date)

        X_accumulated, y_accumulated = [], []
        total_sequences = 0
        total_batches = 0

        for i in range(len(files) - sequence_length):
            input_files = files[i: i + sequence_length]
            target_file = files[i + sequence_length]

            print(f"\n🪟 Sequence #{total_sequences + 1}:")
            print("   🗂️ Inputs:")
            for f in input_files:
                print(f"     - {os.path.basename(f)}")
            print(f"   🎯 Target: {os.path.basename(target_file)}")

            try:
                X_seq = []
                for f in input_files:
                    X_f, _ = self.load_batch([f])  # returns [X], _
                    X_f = X_f[0]
                    if X_f.ndim != 3:
                        raise ValueError(f"Expected (C,H,W), got {X_f.shape}")
                    print(f"     📦 {os.path.basename(f)} | X shape: {X_f.shape}")
                    X_seq.append(X_f)

                X_seq = np.stack(X_seq, axis=0)  # [T, C, H, W]
                print(f"   🔗 Stacked X_seq shape: {X_seq.shape}")

                _, y_target = self.load_batch([target_file])  # returns _, [y]
                y_val = y_target[0]
                print(f"   🎯 y_target shape: {np.array(y_val).shape}")

                X_accumulated.append(X_seq)
                y_accumulated.append(y_val)
                total_sequences += 1

            except Exception as e:
                print(f"   ⚠️ Skipping sequence due to error: {e}")
                continue

            while len(X_accumulated) >= batch_size:
                X_batch = np.stack(X_accumulated[:batch_size], axis=0)  # [B, T, C, H, W]
                y_batch = np.stack(y_accumulated[:batch_size], axis=0)

                print(f"\n📤 Yielding batch #{total_batches + 1}")
                print(f"   ✅ X_batch shape: {X_batch.shape}")
                print(f"   ✅ y_batch shape: {y_batch.shape}")

                yield X_batch, y_batch

                X_accumulated = X_accumulated[batch_size:]
                y_accumulated = y_accumulated[batch_size:]
                total_batches += 1

        if X_accumulated:
            X_batch = np.stack(X_accumulated, axis=0)
            y_batch = np.stack(y_accumulated, axis=0)

            print(f"\n📤 Yielding final (incomplete) batch")
            print(f"   ✅ X_batch shape: {X_batch.shape}")
            print(f"   ✅ y_batch shape: {y_batch.shape}")

            yield X_batch, y_batch

        print("\n✅ Batch generation complete.")
        print(f"   🔢 Total sequences: {total_sequences}")
        print(f"   📦 Total batches: {total_batches + (1 if X_accumulated else 0)}")




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
            self.end_date - timedelta(days=21), 
            self.end_date,
            batch_size=self.batch_size,
        )

    def evaluate_model(
        self, model_file_path, files, start_date, end_date, batch_size=10000
    ):
        if batch_size is not None:
            self.batch_size = batch_size

        print("=" * 60)
        print("🧪 Starting Evaluation Phase")
        print(f"📦 Batch size set to: {self.batch_size}")
        print(f"🗂️ Number of input files: {len(files)}")
        print(f"🕒 Evaluation period: {start_date} to {end_date}")
        print("=" * 60)

        # Load model and set to eval mode
        print("🔄 Loading model from:", model_file_path)
        self.model = self.load_model(model_file_path)
        self.model.eval()
        print(f"✅ Model loaded successfully. Evaluating on device: {self.device}")

        all_y_true = []
        all_y_pred = []
        batch_count = 0
        total_samples = 0

        for X_batch, y_batch in self.load_batch_iterator(files, sequence_length=7, batch_size=self.batch_size):
            print("-" * 60)
            print(f"📊 Processing batch {batch_count + 1}...")
            print(f"  └── Raw X_batch shape: {X_batch.shape}")
            print(f"  └── Raw y_batch length: {len(y_batch)}")

            # Normalize and convert to tensor
            X_batch = self.normalize(X_batch)
            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)

            # Set expected sequence and feature dimensions
            self.seq_len = 7          # number of days (time steps)
            self.num_features = 8     # number of variables per day
            expected_shape = self.seq_len * self.num_features  # 56

            if X_batch.shape[1] != expected_shape:
                raise ValueError(f"❌ Input shape mismatch: got {X_batch.shape}, "
                                f"expected (_, {expected_shape})")

            try:
                X_batch_tensor = X_batch_tensor.view(-1, self.seq_len, self.num_features)
                print(f"  ✅ Reshaped X_batch_tensor: {X_batch_tensor.shape}")
            except Exception as e:
                raise ValueError(f"❌ Error reshaping input: {X_batch_tensor.shape} "
                                f"to (-1, {self.seq_len}, {self.num_features})\n{e}")

            # Run prediction
            try:
                with torch.no_grad():
                    y_pred_batch = self.model(X_batch_tensor).detach().cpu().numpy()
                print(f"  🎯 Predicted batch shape: {y_pred_batch.shape}")
            except Exception as e:
                raise RuntimeError(f"❌ Model prediction failed on batch {batch_count + 1}\n{e}")

            # Aggregate results
            all_y_true.extend(y_batch)
            all_y_pred.extend(y_pred_batch)
            total_samples += len(y_batch)
            batch_count += 1
            print(f"  ✅ Batch {batch_count} complete | Total samples so far: {total_samples}")

        print("=" * 60)
        print("📈 Finalizing Evaluation Metrics")

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        print(f"  🧾 y_true.shape: {y_true.shape}")
        print(f"  🧾 y_pred.shape: {y_pred.shape}")

        # === Metrics ===
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print("✅ Evaluation Complete")
        print(f"  📉 RMSE = {rmse:.4f}")
        print(f"  📊 R2   = {r2:.4f}")
        print("=" * 60)




if __name__ == "__main__":

    # Define start and end date for training
    start_date_str = '2023-10-01'
    end_date_str = '2024-05-15'
    
    hole_model = SNODAS_Transformer_Hole(
        start_date_str = start_date_str,
        end_date_str = end_date_str,
        batch_size = 5000,
        epochs = 2,
        train_ratio = 0.8,
        test_ratio = 0.005,
        val_ratio = 0.01,
        normalization = True,
        retrain = True,
        model_path = "/home/chetana/models/SNODAS_Transformer_Hole_e10_nTrue_seq2seq_latest.model",
        # base_model_path = "/home/chetana/models//SNODAS_Transformer_Hole_e10_nTrue_latest.model"
    )
    
    hole_model.preprocessing(verbose=True, semimonth_only = False)
    # hole_model.train()
    # hole_model.save()
    hole_model.evaluate()
    # hole_model.predict("2024-12-10")
    # hole_model.predict("2024-11-10")
    # hole_model.predict("2024-10-10")
    # hole_model.predict("2024-09-10")
    # hole_model.predict("2024-08-10")
    # hole_model.predict("2024-07-10")
    # hole_model.predict("2024-06-10")
    # hole_model.predict("2024-05-10")
    # hole_model.predict("2024-04-10")
    # hole_model.predict("2024-03-10")
    # hole_model.predict("2024-02-10")
    # hole_model.predict("2024-01-10")


