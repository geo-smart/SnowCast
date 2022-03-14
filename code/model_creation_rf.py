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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from base_hole import *

class RandomForestHole(BaseHole):

  def prepare_training():
    pass
  
  def get_model():
    rfc_pipeline = Pipeline(steps = [
      ('data_scaling', StandardScaler()),
      ('model', RandomForestClassifier(max_depth = 10,
                                       min_samples_leaf = 3,
                                       min_samples_split = 4,
                                       n_estimators = 200))])
    return rfc_pipeline


