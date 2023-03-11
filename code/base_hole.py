'''
The wrapper for all the snowcast_wormhole predictors
'''
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

class BaseHole:
  
  all_ready_file = f"{github_dir}/data/ready_for_training/all_ready_new.csv"
  
  def __init__(self):
    self.classifier = self.get_model()
    self.holename = self.__class__.__name__ 
    self.train_x = None
    self.train_y = None
    self.test_x = None
    self.test_y = None
    self.test_y_results = None
    self.save_file = None
    
  def save(self):
    now = datetime.now()
    date_time = now.strftime("%Y%d%m%H%M%S")
    self.save_file = f"{github_dir}/model/wormhole_{self.holename}_{date_time}.joblib"
    print(f"Saving model to {self.save_file}")
    joblib.dump(self.classifier, self.save_file)
  
  def preprocessing(self):
    all_ready_pd = pd.read_csv(self.all_ready_file, header=0, index_col=0)
    input_columns = ["year", "m", "day", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "lat", "lon", "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe_0719", "depth_0719"]
    
    all_cols = input_columns
    all_cols.append("swe_snotel")
    print("all columns: ", all_cols)
    print(type(i) for i in all_cols)
    all_ready_pd = all_ready_pd[all_cols]
#     all_ready_pd = all_ready_pd.fillna(10000) # replace all nan with 10000
    all_ready_pd = all_ready_pd[all_ready_pd["swe_snotel"]!=-1]
    all_ready_pd = all_ready_pd.dropna()
    print("all ready df columns used for traing: ",all_ready_pd.columns)
    print("all ready df columns shape: ",all_ready_pd.shape)
    train, test = train_test_split(all_ready_pd, test_size=0.2)
#     "cell_id", "year", "m", "day", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx","vpd", "vs", "lat", "lon",
#                  "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe_0719", "depth_0719", "swe_snotel"
    print("Training data columns: ",train.columns)
    self.train_x, self.train_y = train[input_columns].to_numpy().astype('float'), train[['swe_snotel']].to_numpy().astype('float')
    self.test_x, self.test_y = test[input_columns].to_numpy().astype('float'), test[['swe_snotel']].to_numpy().astype('float')
  
  def train(self):
    self.classifier.fit(self.train_x, self.train_y)
  
  def test(self):
    self.test_y_results = self.classifier.predict(self.test_x)
    return self.test_y_results
  
  def predict(self, input_x):
    return self.classifier.predict(input_x)
  
  def evaluate(self):
    pass
  
  def get_model(self):
    pass
  
  def post_processing(self):
    pass
