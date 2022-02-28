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

#pd.set_option('display.max_columns', None)

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
modis_ready_file = f"{github_dir}/data/ready_for_training/modis_train_ready.csv"
modis_ready_pd = pd.read_csv(modis_ready_file, header=0, index_col=0)

pd_to_clean = modis_ready_pd[["year", "m", "doy", "ndsi", "swe"]].dropna()

all_features = pd_to_clean[["year", "m", "doy", "ndsi"]].to_numpy()
all_labels = pd_to_clean[["swe"]].to_numpy().ravel()

train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels, test_size=0.20, random_state=42)

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

print("==> create random forest model")

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

randomForestregModel = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = randomForestregModel, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(train_features, train_labels)

print("rf_random.best_params_: ", rf_random.best_params_)

def evaluate(model, test_features, y_test, model_name):
    y_predicted = model.predict(test_features)
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    r2 = metrics.r2_score(y_test, y_predicted)
    rmse = math.sqrt(mse)

    print("The model performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))
    print('RMSE is {}'.format(rmse))
    
    return r2

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels, "Base Model")

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels, "Optimized")

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

os.makedirs(f"{github_dir}/model/", exist_ok=True)
# save
joblib.dump(base_model, f"{github_dir}/model/wormhole_random_forest_basic_v2.joblib")
joblib.dump(best_random, f"{github_dir}/model/wormhole_random_forest_v2.joblib")
print("wormhole_random_forest is saved to file")

