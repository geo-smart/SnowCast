from sklearn.ensemble import RandomForestRegressor
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
import math
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from base_hole import BaseHole
from sklearn.model_selection import train_test_split
from datetime import datetime

homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

class RandomForestHole(BaseHole):
  
  all_ready_file = f"{github_dir}/data/ready_for_training/all_ready.csv"
  

  def preprocessing(self):
    all_ready_pd = pd.read_csv(self.all_ready_file, header=0, index_col=0)
    all_ready_pd = all_ready_pd.fillna(10000) # replace all nan with 10000
    train, test = train_test_split(all_ready_pd, test_size=0.2)
    self.train_x, self.train_y = train[["year", "m", "doy", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "lat", "lon",
                 "elevation", "aspect", "curvature", "slope", "eastness", "northness"]].to_numpy().astype('float'), train[['swe','depth']].to_numpy().astype('float')
    self.test_x, self.test_y = test[["year", "m", "doy", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "lat", "lon",
                 "elevation", "aspect", "curvature", "slope", "eastness", "northness"]].to_numpy().astype('float'), test[['swe','depth']].to_numpy().astype('float')
  
  def get_model(self):
    rfc_pipeline = Pipeline(steps = [
      ('data_scaling', StandardScaler()),
      ('model', RandomForestRegressor(max_depth = 15,
                                       min_samples_leaf = 0.004,
                                       min_samples_split = 0.008,
                                       n_estimators = 25))])
    return rfc_pipeline

  def evaluate(self):
    mae = metrics.mean_absolute_error(self.test_y, self.test_y_results)
    mse = metrics.mean_squared_error(self.test_y, self.test_y_results)
    r2 = metrics.r2_score(self.test_y, self.test_y_results)
    rmse = math.sqrt(mse)

    print("The random forest model performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))
    print('RMSE is {}'.format(rmse))
    return {"mae":mae, "mse": mse, "r2": r2, "rmse": rmse}
