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
#print(station_cell_mapper_pd.head())

#example_mod_pd = pd.read_csv(example_mod_file, header=0, index_col=0)
#print(example_mod_pd.shape)


def integrate_modis():
  """
  Integrate all MODIS data into mod_all.csv
  """
  all_mod_file = f"{github_dir}/data/ready_for_training/modis_all.csv"
  if os.path.isfile(all_mod_file):
    return
  dates = pd.date_range(start='1/1/2013', end='12/31/2021', freq='D').astype(str)
  mod_all_df = pd.DataFrame(columns=["date"])
  mod_all_df['date'] = dates
  
  #print(mod_all_df.head())
  for ind in station_cell_mapper_pd.index:
    current_cell_id = station_cell_mapper_pd["cell_id"][ind]
    print(current_cell_id)
    mod_single_file = f"{github_dir}/data/modis/mod10a1_ndsi_{current_cell_id}.csv"
    if os.path.isfile(mod_single_file):
      mod_single_pd = pd.read_csv(mod_single_file, header=0)
      mod_single_pd = mod_single_pd[["date", "mod10a1_ndsi"]]
      mod_single_pd = mod_single_pd.rename(columns={"mod10a1_ndsi": current_cell_id})
      mod_single_pd['date'] = pd.to_datetime(mod_single_pd['date']).astype(str)
      print(mod_all_df.shape)
      mod_all_df = pd.merge(mod_all_df, mod_single_pd, how='left', on="date")
  mod_all_df.to_csv(all_mod_file)

  
def integrate_sentinel1():
  """
  Integrate all Sentinel 1 data into sentinel1_all.csv
  """
  all_sentinel1_file = f"{github_dir}/data/ready_for_training/sentinel1_all.csv"
  if os.path.isfile(all_mod_file):
    return
  dates = pd.date_range(start='1/1/2013', end='12/31/2021', freq='D').astype(str)
  sentinel1_all_df = pd.DataFrame(columns=["date"])
  sentinel1_all_df['date'] = dates
  #print(mod_all_df.head())

  def getDateStr(x):
    return x.split(" ")[0]

  for ind in station_cell_mapper_pd.index:
    current_cell_id = station_cell_mapper_pd["cell_id"][ind]
    print(current_cell_id)
    sentinel1_single_file = f"{github_dir}/data/sentinel1/s1_grd_vv_{current_cell_id}.csv"
    if os.path.isfile(sentinel1_single_file) and current_cell_id not in sentinel1_all_df :
      sentinel1_single_pd = pd.read_csv(sentinel1_single_file, header=0)
      sentinel1_single_pd = sentinel1_single_pd[["date", "s1_grd_vv"]]
      sentinel1_single_pd = sentinel1_single_pd.rename(columns={"s1_grd_vv": current_cell_id})
      #sentinel1_single_pd['date'] = sentinel1_single_pd['date'].astype('datetime64[ns]')
      sentinel1_single_pd['date'] = pd.to_datetime(sentinel1_single_pd['date']).dt.round("D").astype(str)
      print("sentinel1_single_pd: ", sentinel1_single_pd.head())
      print("sentinel1_single_pd check value: ", sentinel1_single_pd[sentinel1_single_pd["date"]=="2015-04-01"])
      sentinel1_single_pd = sentinel1_single_pd.drop_duplicates(subset=['date'], keep='first') # this will remove all the other values of the same day
      
      sentinel1_all_df = pd.merge(sentinel1_all_df, sentinel1_single_pd, how='left', on="date")
      print("sentinel1_all_df check value: ", sentinel1_all_df[sentinel1_all_df["date"]=="2015-04-01"])
      print("sentinel1_all_df: ", sentinel1_all_df.shape)
      

  print(sentinel1_all_df.shape)
  sentinel1_all_df.to_csv(all_sentinel1_file)

def integrate_gridmet():
  pass
  
  
def prepare_training_csv():
  """
  MOD model:
    input columns: [m, doy, ndsi]
    output column: [swe]
  Sentinel1 model:
    input columns: [m, doy, grd]
    output column: [swe]
  """
  all_mod_file = f"{github_dir}/data/ready_for_training/modis_all.csv"
  modis_all_pd = pd.read_csv(all_mod_file, header=0)
  print("modis_all_size: ", modis_all_pd.shape)
  print("station size: ", station_cell_mapper_pd.shape)
  print("training_feature_pd size: ", training_feature_pd.shape)
  print("testing_feature_pd size: ", testing_feature_pd.shape)
  
  ndsi_training_pd = pd.DataFrame(columns=["year", "m", "doy", "ndsi", "swe"])
  modis_all_pd = modis_all_pd.reset_index()
  for index, row in modis_all_pd.iterrows():
    dt = datetime.strptime(row['date'], '%Y-%m-%d')
    month = dt.month
    year = dt.year
    doy = dt.timetuple().tm_yday
    for i in range(3,len(row.index)):
      cell_id = row.index[i][:-2]
      ndsi = row.values[i]
      if cell_id in train_labels_pd.index and row['date'] in train_labels_pd:
        swe = train_labels_pd.loc[cell_id, row['date']]
        if not np.isnan(swe):
          ndsi_training_pd.loc[len(ndsi_training_pd.index)] = [year, month, doy, ndsi, swe]
  
  print(ndsi_training_pd.shape)
  ndsi_training_pd.to_csv(f"{github_dir}/data/ready_for_training/modis_ready.csv")
  
  all_sentinel1_file = f"{github_dir}/data/ready_for_training/sentinel1_all.csv"
  sentinel1_all_pd = pd.read_csv(all_sentinel1_file, header=0)
  grd_all_pd = pd.DataFrame(columns=["year", "m", "doy", "grd", "swe"])
  grd_all_pd = grd_all_pd.reset_index()
  for index, row in sentinel1_all_pd.iterrows():
    dt = datetime.strptime(row['date'], '%Y-%m-%d')
    year = dt.year
    month = dt.month
    doy = dt.timetuple().tm_yday
    for i in range(3,len(row.index)):
      cell_id = row.index[i]
      grd = row.values[i]
      if not np.isnan(grd) and cell_id in train_labels_pd.index and row['date'] in train_labels_pd:
        swe = train_labels_pd.loc[cell_id, row['date']]
        if not np.isnan(swe):
          print([month, doy, grd, swe])
          grd_all_pd = grd_all_pd.append({"year": year, "m":month, "doy": doy, "grd": grd, "swe": swe}, ignore_index = True)
  
  print(grd_all_pd.shape)
  grd_all_pd.to_csv(f"{github_dir}/data/ready_for_training/sentinel1_ready.csv")
  
exit() # done already

integrate_modis()
integrate_sentinel1()
integrate_gridmet()
prepare_training_csv()


  
  
  

