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
from model_creation_rf import RandomForestHole


class XGBoostHole(RandomForestHole):

  def get_model(self):
    rfc_pipeline = Pipeline(steps = [
      ('data_scaling', StandardScaler()),
      ('model', RandomForestRegressor(max_depth = 15,
                                       min_samples_leaf = 0.004,
                                       min_samples_split = 0.008,
                                       n_estimators = 25))])
    return rfc_pipeline



