# Integrate all the datasets into one training dataset
import json
import pandas as pd
import ee
import seaborn as sns
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import geojson
import numpy as np
import os.path
import math
from datetime import datetime


print("integrating datasets into one dataset")
#pd.set_option('display.max_columns', None)

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
gridcells_file = f"{github_dir}/data/snowcast_provided/grid_cells.geojson"
model_dir = f"{github_dir}/model/"
training_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_train_features.csv"
testing_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_test_features.csv"
train_labels_file = f"{github_dir}/data/snowcast_provided/train_labels.csv"
ground_measure_metadata_file = f"{github_dir}/data/snowcast_provided/ground_measures_metadata.csv"
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"

#example_mod_file = f"{github_dir}/data/modis/mod10a1_ndsi_f191fe19-0e81-4bc9-9980-29738a05a49b.csv"


training_feature_pd = pd.read_csv(training_feature_file, header=0, index_col=0)
testing_feature_pd = pd.read_csv(testing_feature_file, header=0, index_col=0)
train_labels_pd = pd.read_csv(train_labels_file, header=0, index_col=0)
#print(training_feature_pd.head())

station_cell_mapper_pd = pd.read_csv(station_cell_mapper_file, header=0, index_col=0)

ndsi_testing_pd = pd.DataFrame(columns=["year", "m", "doy", "ndsi", "swe"])
  modis_all_pd = modis_all_pd.reset_index()
  for index, row in modis_all_pd.iterrows():
    dt = datetime.strptime(row['date'], '%Y-%m-%d')
    month = dt.month
    year = dt.year
    doy = dt.timetuple().tm_yday
    for i in range(3,len(row.index)):
      cell_id = row.index[i][:-2]
      ndsi = row.values[i]
      if cell_id in test_labels_pd.index and row['date'] in train_labels_pd:
        swe = train_labels_pd.loc[cell_id, row['date']]
        if not np.isnan(swe):
          ndsi_training_pd.loc[len(ndsi_training_pd.index)] = [year, month, doy, ndsi, swe]
