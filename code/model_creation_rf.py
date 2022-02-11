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
home = str(Path.home())


print("==> create random forest model")

randomForestregModel = RandomForestRegressor(max_depth=15)

os.makedirs(f"{home}/model/", exist_ok=True)

# save
joblib.dump(randomForestregModel, f"{home}/model/wormhole_random_forest.joblib")

print("wormhole_random_forest is saved to file")

