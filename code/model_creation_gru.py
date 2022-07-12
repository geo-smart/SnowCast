import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
from sklearn import tree
import joblib
import os
from pathlib import Path
import json
import geopandas as gpd
import geojson
import os.path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from base_hole import BaseHole
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras import optimizers

homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

class GRU_Model:
  
    all_ready_file = f"{github_dir}/data/ready_for_training/all_ready.csv"
  
    def preprocessing(self):
        all_ready_file = f"{github_dir}/data/ready_for_training/all_ready.csv"
        all_ready_pd = pd.read_csv(all_ready_file, header=0, index_col=0)
        all_ready_pd = all_ready_pd.fillna(10000) # replace all nan with 10000
        
        train, test = train_test_split(all_ready_pd, test_size=0.2)
        self.X_train, self.y_train = train[['year','m','doy','ndsi','grd','eto','pr','rmax','rmin','tmmn','tmmx','vpd','vs','lat','lon','elevation','aspect','curvature','slope','eastness','northness']].to_numpy().astype('float'), train['swe'].to_numpy().astype('float')
       
        self.X_test, self.y_test = test[['year','m','doy','ndsi','grd','eto','pr','rmax','rmin','tmmn','tmmx','vpd','vs','lat','lon','elevation','aspect','curvature','slope','eastness','northness']].to_numpy().astype('float'), test['swe'].to_numpy().astype('float')
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        
        self.X_train = np.reshape(self.X_train, (45389, 21, 1))
        self.X_test = np.reshape(self.X_test, (11348, 21, 1))        
    
    def train(self):
        # Model Creation
        model = Sequential()
        model.add(GRU(128, input_shape=(self.X_train.shape[1:]), activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(GRU(128, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1))
        
        # Model Compilation
        model.compile(optimizer=optimizers.Adam(lr=0.0001),
                loss='mse',
                metrics=['mae'])
        
        # Model Fitting
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=64, epochs=1)
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        def evaluate(self):
            mae = model.history['mae']
            mse = model.history['loss']
            r2 = metrics.r2_score(self.test_y, self.test_y_results)
            rmse = math.sqrt(mse)

            print("The LSTM model performance for testing set")
            print("--------------------------------------")
            print('MAE is {}'.format(mae))
            print('MSE is {}'.format(mse))
            return {"mae":mae, "mse": mse}



