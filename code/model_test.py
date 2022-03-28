# Test models

# Random Forest model creation and save to file

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
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

def turn_doy_to_date(year, doy):
  doy = int(doy)
  doy = str(doy)
  doy.rjust(3 + len(doy), '0')
  #res = datetime.strptime(str(year) + "-" + doy, "%Y-%j").strftime("%m/%d/%Y")
  res = datetime.strptime(str(year) + "-" + doy, "%Y-%j").strftime("%Y-%m-%d")
  return res

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
test_ready_file = f"{github_dir}/data/ready_for_testing/all_ready.csv"
test_ready_pd = pd.read_csv(test_ready_file, header=0, index_col=0)
submission_file = f"{github_dir}/data/snowcast_provided/submission_format.csv"
submission_pd = pd.read_csv(submission_file, header=0, index_col=0)
predicted_file = f"{homedir}/Documents/GitHub/SnowCast/data/results/wormhole_output.csv"

train_cols = ['year','m','doy','ndsi','grd','eto','pr','rmax','rmin','tmmn','tmmx','vpd','vs','lat','lon','elevation','aspect','curvature','slope','eastness','northness']

print(test_ready_pd.shape)
pd_to_clean = test_ready_pd[train_cols]
print("PD shape: ", pd_to_clean.shape)

doy_list = test_ready_pd["doy"].unique()
print(doy_list)

date_list = [turn_doy_to_date(2022, doy_list[i]) for i in range(len(doy_list)) ]
print(date_list)

all_features = pd_to_clean.to_numpy()
print("train feature shape: ", all_features.shape)
#all_features = pd_to_clean[["year", "m", "doy", "ndsi"]].to_numpy()
#all_labels = pd_to_clean[["swe"]].to_numpy().ravel()

#base_model = joblib.load(f"{homedir}/Documents/GitHub/snowcast_trained_model/model/wormhole_random_forest_basic_v2.joblib")
#base_model = joblib.load(f"{homedir}/Documents/GitHub/snowcast_trained_model/model/wormhole_random_forest_basic_v2.joblib")

best_random = joblib.load(f"{homedir}/Documents/GitHub/SnowCast/model/wormhole_20222703183552.joblib")
y_predicted = best_random.predict(all_features)
print(y_predicted) #first got daily prediction

target_dates = ["2022-01-13","2022-01-20","2022-01-27","2022-02-03","2022-02-10","2022-02-17","2022-02-24","2022-03-03","2022-03-10","2022-03-17","2022-03-24","2022-03-31","2022-04-07","2022-04-14","2022-04-21","2022-04-28","2022-05-05","2022-05-12","2022-05-19","2022-05-26","2022-06-02","2022-06-09","2022-06-16","2022-06-23","2022-06-30"]
print("taregt date list: ", len(target_dates))

daily_predictions = pd.DataFrame(columns = target_dates, index = submission_pd.index)
for i in range(len(y_predicted)):
  doy = all_features[i][2]
  #print(doy)
  ndate = turn_doy_to_date(2022, doy)
  if ndate in target_dates:
    cell_id = test_ready_pd["cell_id"].iloc[i]
    daily_predictions.at[cell_id, ndate] = y_predicted[i]
  #print(ndate, cell_id)
  #print(y_predicted[i])
  
print(daily_predictions.shape)
#daily_predictions = daily_predictions[["2022-01-13"]]

if os.path.exists(predicted_file):
  os.remove(predicted_file)
daily_predictions.to_csv(predicted_file)


# turn daily into weekly using mean values






