import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from snowcast_utils import work_dir, month_to_season
from datetime import datetime
import os

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, \
    Conv1D, MaxPooling1D, Flatten, Input, Add, \
    LayerNormalization, Embedding, Layer, InputSpec, Lambda, Reshape, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import optuna
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import torch

from transformers import TFBertModel, TFGPT2Model, TFT5Model, BertTokenizer, GPT2Tokenizer, T5Tokenizer
# from spektral.layers import GraphConv, GlobalSumPool
from spektral.layers import GCNConv, GlobalSumPool
# from neural_ode import NeuralODE  # Assuming you have a custom implementation or library for NeuralODE
from torchdiffeq import odeint
import torch.nn as nn

# Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()

working_dir = work_dir

homedir = os.path.expanduser('~')
now = datetime.now()
date_time = now.strftime("%Y%d%m%H%M%S")

github_dir = f"{homedir}/Documents/GitHub/SnowCast"
training_data_path = f"{working_dir}/snotel_ghcnd_stations_4yrs_all_cols_log10_subset.csv"

model_save_file = f"{github_dir}/model/wormhole_autosai_{date_time}.keras"

def get_data():
    # Read the data from the CSV file
    print(f"start to read data {training_data_path}")
    df = pd.read_csv(training_data_path)
    df.dropna(inplace=True)

    print(df.head())
    print(df.columns)
    print(df.shape)

    # print("randomly sample 5000 rows")
    # df = df.sample(n=5000, random_state=1)  # random_state is optional but useful for reproducibility
    print(df.shape)


    # Load and prepare your data
    # data = pd.read_csv('your_dataset.csv')  # Replace with your actual data source
    X = df.drop(columns=['swe_value', 'date', 'station_name',])  # Replace with your actual target column
    y = df['swe_value']

    print("Encode categorical features")

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    print("Split the data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_val = y_val.to_numpy().reshape(-1, 1)
    print("Scale the data")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val 

# Define the Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim)
        )

    def forward(self, t, y):
        return self.net(y)

class NeuralODEModel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralODEModel, self).__init__()
        self.odefunc = ODEFunc(input_dim)
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_time)
        return out[1]


class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = 'he_normal'

    def build(self, input_shape):
        self.input_dim_capsule = input_shape[-1]
        self.W = self.add_weight(shape=[self.num_capsule, input_shape[1], self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        b = K.zeros_like(inputs_hat[:, :, :, 0])

        for i in range(self.routings):
            c = K.softmax(b, 1)
            outputs = K.batch_dot(c, inputs_hat, [2, 2])
            if i < self.routings - 1:
                b = K.batch_dot(outputs, inputs_hat, [2, 3])
        
        return K.sqrt(K.sum(K.square(outputs), 2))

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

class Length(Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(Layer):
    def call(self, inputs, mask=None):
        if mask is None:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.expand_dims(K.argmax(x, 1), -1)
        return K.batch_dot(inputs, mask, [1, 1])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

def create_capsnet_model(input_shape):
    inputs = Input(shape=(input_shape,))
    primary_caps = Dense(128, activation='relu')(inputs)
    primary_caps = Reshape((16, 8))(primary_caps)
    digit_caps = CapsuleLayer(num_capsule=10, dim_capsule=16, routings=3)(primary_caps)
    outputs = Length()(digit_caps)
    model = Model(inputs, outputs)
    return model

# Define the function to create a diverse Keras model
def create_model(trial, X_train, X_val, y_train, y_val):
    input_shape = X_train.shape[1]
    model_type = trial.suggest_categorical('model_type', [
        'dense', 'cnn', 'lstm', 'transformer', 
        'tabnet', 'random_forest', 'xgboost', 'lightgbm',
        # 'bert', 'gpt', 't5', 
        # 'gcn', 
        # 'gat', 
        # 'vae', 
        'gru', 
        'bilstm', 
        # 'neural_ode', 
        # 'capsnet'
        ])

    if model_type in ['bert', 'gpt', 't5']:
        # Convert numerical data to strings for text-based models
        X_train = X_train.astype(str)
        X_val = X_val.astype(str)
    else:
        # Convert numerical data to strings for text-based models
        X_train = X_train.astype(float)
        X_val = X_val.astype(float)

    print(f"--- > This chosen model is {model_type}")

    if model_type == 'dense':
        model = Sequential()
        num_layers = trial.suggest_int('num_layers', 1, 5)
        for i in range(num_layers):
            num_units = trial.suggest_int(f'num_units_l{i}', 16, 128, log=True)
            if i == 0:
                model.add(Dense(num_units, activation='relu', input_shape=(input_shape,)))
            else:
                model.add(Dense(num_units, activation='relu'))
            dropout_rate = trial.suggest_float(f'dropout_rate_l{i}', 0.1, 0.5)
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        learning_rate = trial.suggest_float('learning_rate_dense', 1e-5, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    elif model_type == 'cnn':
        model = Sequential()
        model.add(Conv1D(filters=trial.suggest_int('filters', 16, 64, log=True),
                        kernel_size=trial.suggest_int('kernel_size', 3, 5),
                        activation='relu',
                        input_shape=(input_shape, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        learning_rate = trial.suggest_float('learning_rate_cnn', 1e-5, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    elif model_type == 'lstm':
        model = Sequential()
        model.add(LSTM(units=trial.suggest_int('lstm_units', 16, 64, log=True),
                    input_shape=(input_shape, 1)))
        model.add(Dense(1, activation='linear'))
        learning_rate = trial.suggest_float('learning_rate_lstm', 1e-5, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    elif model_type == 'transformer':
        inputs = Input(shape=(input_shape, 1))
        attention = MultiHeadAttention(num_heads=trial.suggest_int('num_heads', 2, 8), key_dim=trial.suggest_int('key_dim', 16, 64))(inputs, inputs)
        attention = Add()([inputs, attention])
        attention = LayerNormalization(epsilon=1e-6)(attention)
        outputs = Flatten()(attention)
        outputs = Dense(1, activation='linear')(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        learning_rate = trial.suggest_float('learning_rate_transformer', 1e-5, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    elif model_type == 'tabnet':
        tabnet_model = TabNetRegressor(
            n_d=trial.suggest_int('n_d', 8, 64),
            n_a=trial.suggest_int('n_a', 8, 64),
            n_steps=trial.suggest_int('n_steps', 3, 10),
            gamma=trial.suggest_float('gamma', 1.0, 2.0),
            lambda_sparse=trial.suggest_float('lambda_sparse', 1e-6, 1e-3)
        )
        tabnet_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
        )
        return tabnet_model
    
    elif model_type == 'random_forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model
    
    elif model_type == 'xgboost':
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': trial.suggest_float('learning_rate_xgboost', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eval_metric': 'mae',
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model
    
    elif model_type == 'lightgbm':
        params = {
            'objective': 'regression',
            'learning_rate': trial.suggest_float('learning_rate_lightgbm', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return model

    elif model_type == 'gru':
        model = Sequential()
        model.add(GRU(units=trial.suggest_int('gru_units', 16, 64, log=True), input_shape=(input_shape, 1)))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'bilstm':
        model = Sequential()
        model.add(Bidirectional(LSTM(units=trial.suggest_int('bilstm_units', 16, 64, log=True)), input_shape=(input_shape, 1)))
        model.add(Dense(1, activation='linear'))
    
    elif model_type == 'vae':
        # Define VAE architecture
        inputs = Input(shape=(input_shape,))
        h = Dense(trial.suggest_int('vae_units', 64, 128), activation='relu')(inputs)
        z_mean = Dense(trial.suggest_int('z_mean', 2, 10))(h)
        z_log_var = Dense(trial.suggest_int('z_log_var', 2, 10))(h)
        z = Lambda(sampling, output_shape=(trial.suggest_int('z_dim', 2, 10),))([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        latent_inputs = Input(shape=(trial.suggest_int('z_dim', 2, 10),), name='z_sampling')
        h_decoded = Dense(trial.suggest_int('vae_units', 64, 128), activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(h_decoded)
        decoder = Model(latent_inputs, outputs, name='decoder')
        outputs = decoder(encoder(inputs)[2])
        model = Model(inputs, outputs, name='vae_mlp')
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_shape
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        model.add_loss(vae_loss)
    
    elif model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')
        if isinstance(X_train[0], str):
            inputs = tokenizer(X_train.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=128)
        else:
            raise ValueError("Input data must be of type str for BERT model")
        outputs = model(inputs)
        pooled_output = outputs[0][:, -1]
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=pooled_output.shape[1:]))
    
    elif model_type == 'gpt':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = TFGPT2Model.from_pretrained('gpt2')
        if isinstance(X_train[0], str):
            inputs = tokenizer(X_train.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=128)
        else:
            raise ValueError("Input data must be of type str for GPT model")
        outputs = model(inputs)
        pooled_output = outputs[0][:, -1]
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=pooled_output.shape[1:]))
    
    elif model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = TFT5Model.from_pretrained('t5-small')
        if isinstance(X_train[0], str):
            inputs = tokenizer(X_train.tolist(), return_tensors='tf', padding=True, truncation=True, max_length=128)
        else:
            raise ValueError("Input data must be of type str for T5 model")
        outputs = model(inputs)
        pooled_output = outputs[0][:, -1]
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=pooled_output.shape[1:]))
    
    elif model_type == 'gcn':
        A = trial.suggest_categorical('A', [np.eye(input_shape), np.random.rand(input_shape, input_shape)])
        X_input = Input(shape=(input_shape,))
        H = GCNConv(32, activation='relu')([X_input, A])
        H = GlobalSumPool()(H)
        outputs = Dense(1)(H)
        model = Model(inputs=X_input, outputs=outputs)
    
    elif model_type == 'gat':
        A = trial.suggest_categorical('A', [np.eye(input_shape), np.random.rand(input_shape, input_shape)])
        X_input = Input(shape=(input_shape,))
        H = GCNConv(32, activation='relu')([X_input, A])
        H = GlobalSumPool()(H)
        outputs = Dense(1)(H)
        model = Model(inputs=X_input, outputs=outputs)
    
    elif model_type == 'neural_ode':
        model = NeuralODEModel(input_dim=input_shape)
    
    elif model_type == 'capsnet':
        model = create_capsnet_model(input_shape)

    # Only for Keras models
    if model_type in ['dense', 'cnn', 'lstm', 'transformer', 
        'bert', 'gpt', 't5', 'gcn', 'gat', 'vae', 
        'gru', 'bilstm', 'neural_ode', 'capsnet']:
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model

# Define the objective function for Optuna
def objective(trial):
    X_subset = X_train
    y_subset = y_train

    model = create_model(trial, X_train, X_val, y_train, y_val)

    if isinstance(model, TabNetRegressor):
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
    elif isinstance(model, (RandomForestRegressor, xgb.XGBRegressor, 
        lgb.LGBMRegressor)):
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
    else:
        # all tensorflow models
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        model.fit(X_subset, y_subset, 
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            epochs=100, 
            batch_size=256, 
            verbose=0)
        loss, mae = model.evaluate(X_val, y_val, verbose=0)


    return mae

def do_trials(X_train, X_val, y_train, y_val ):
    
    print("If using CNN, LSTM, or Transformer, add a new dimension for channels")
    # X_train = np.expand_dims(X_train, axis=-1)
    # X_val = np.expand_dims(X_val, axis=-1)
    # Convert y_train and y_val to NumPy arrays
    
    print("y_train.shape = ", y_train.shape)

    print("Run the Bayesian optimization")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Get the best hyperparameters")
    best_params = study.best_params
    print(f"********* Best hyperparameters: {best_params}")

    print("Train the best model on the full dataset")
    best_model = create_model(study.best_trial, X_train, X_val, y_train, y_val)
    if isinstance(best_model, TabNetRegressor):
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
        )
        best_model.save_model('best_tabnet_model')
        print("Best TabNet model saved as best_tabnet_model.zip")
    else:
        best_model.fit(X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=50,  
            batch_size=256, 
            verbose=0)
        best_model.save(model_save_file)
        print(f"Best model saved as {model_save_file}")

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = get_data()
    for _ in range(5):
        do_trials(X_train, X_val, y_train, y_val)


