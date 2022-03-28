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

from datetime import date

pd.set_option('display.max_columns', None)

today = date.today()

# dd/mm/YY
start_date = "2022-01-01"
#end_date = today.strftime("%Y-%m-%d")
end_date = "2022-03-20"
print("d1 =", end_date)

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
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format.csv"

#example_mod_file = f"{github_dir}/data/modis/mod10a1_ndsi_f191fe19-0e81-4bc9-9980-29738a05a49b.csv"


training_feature_pd = pd.read_csv(training_feature_file, header=0, index_col=0)
testing_feature_pd = pd.read_csv(testing_feature_file, header=0, index_col=0)
train_labels_pd = pd.read_csv(train_labels_file, header=0, index_col=0)
submission_format_pd = pd.read_csv(submission_format_file, header=0, index_col=0)
#print(training_feature_pd.head())

station_cell_mapper_pd = pd.read_csv(station_cell_mapper_file, header=0, index_col=0)
#print(station_cell_mapper_pd.head())

#example_mod_pd = pd.read_csv(example_mod_file, header=0, index_col=0)
#print(example_mod_pd.shape)
def getDateStr(x):
  return x.split(" ")[0]

def integrate_modis():
  """
  Integrate all MODIS data into mod_all.csv
  """
  all_mod_file = f"{github_dir}/data/ready_for_testing/modis_all.csv"
  ready_mod_file = f"{github_dir}/data/sat_testing/modis/mod10a1_ndsi_{start_date}_{end_date}.csv"
  if os.path.exists(all_mod_file):
    os.remove(all_mod_file)
  old_modis_pd = pd.read_csv(ready_mod_file, header = 0)
  old_modis_pd = old_modis_pd.drop(columns=['date'])
  old_modis_pd.rename(columns = {'Unnamed: 0':'date'}, inplace = True)
  #cell_id_list = old_modis_pd["cell_id"].unique()
  #cell_id_list = np.insert(cell_id_list, 0, "data")
  cell_id_list = submission_format_pd.index
  date_list = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)
  
  rows = date_list
  cols = cell_id_list
  new_modis_pd = pd.DataFrame(([0.0 for col in cols] for row in rows), index=rows, columns=cols)
  
  for i, row in old_modis_pd.iterrows():
    cdate = row['date']
    ndsi = row['mod10a1_ndsi']
    cellid = row['cell_id']
    #print(f"{cdate} - {ndsi} - {cellid}")
    new_modis_pd.at[cdate, cellid] = ndsi
  
  #modis_np = numpy.zeros((len(date_list), len(cell_id_list)+1))
  #modis_np[0] = cell_id_list
  
  #s1_pd.loc[:, ~s1_pd.columns.str.match('Unnamed')]
  #print(new_modis_pd.head())
  new_modis_pd.to_csv(all_mod_file)

  
def integrate_sentinel1():
  """
  Integrate all Sentinel 1 data into sentinel1_all.csv
  Turn the rows into "daily", right now it has datetime stamps.
  """
  all_sentinel1_file = f"{github_dir}/data/ready_for_testing/sentinel1_all.csv"
  ready_sentinel1_file = f"{github_dir}/data/sat_testing/sentinel1/s1_grd_vv_{start_date}_{end_date}.csv"
  if os.path.exists(all_sentinel1_file):
    os.remove(all_sentinel1_file)
  old_s1_pd = pd.read_csv(ready_sentinel1_file, header = 0)
  old_s1_pd = old_s1_pd.drop(columns=['date'])
  old_s1_pd.rename(columns = {'Unnamed: 0':'date'}, inplace = True)
  #s1_pd.loc[:, ~s1_pd.columns.str.match('Unnamed')]
  
  #cell_id_list = old_s1_pd["cell_id"].unique()
  cell_id_list = submission_format_pd.index
  #date_list = old_s1_pd["date"].unique()
  date_list = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)
  
  
  rows = date_list
  cols = cell_id_list
  new_s1_pd = pd.DataFrame(([0.0 for col in cols] for row in rows), index=rows, columns=cols)
  
  for i, row in old_s1_pd.iterrows():
    cdate = row['date']
    xdate = datetime.strptime(cdate, "%Y-%m-%d %H:%M:%S") #3/7/2022  2:00:48 AM
    sdate = xdate.strftime("%Y-%m-%d")
    grd = row['s1_grd_vv']
    cellid = row['cell_id']
    new_s1_pd.at[sdate, cellid] = float(grd)
  
  new_s1_pd.to_csv(all_sentinel1_file)

def integrate_gridmet():
  """
  Integrate all gridMET data into gridmet_all.csv
  """
  
  dates = pd.date_range(start=start_date, end=end_date, freq='D').astype(str)
  
  #print(mod_all_df.head())
  var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']
  
  for var in var_list:
    print("Processing ", var)
    gridmet_all_df = pd.DataFrame(columns=["date"])
    gridmet_all_df['date'] = dates
    all_gridmet_file = f"{github_dir}/data/ready_for_testing/gridmet_{var}_all.csv"
    if os.path.exists(all_gridmet_file):
      print("skipping")
      continue
      #os.remove(all_gridmet_file)
    cell_num = 0
    for current_cell_id in submission_format_pd.index:
      cell_num += 1
      print("current_cell_id:", cell_num)
      gridmet_single_file = f"{github_dir}/data/sim_testing/gridmet/{var}_{current_cell_id}.csv"
      if os.path.isfile(gridmet_single_file) and current_cell_id not in gridmet_all_df :
        gridmet_single_pd = pd.read_csv(gridmet_single_file, header=0)
        gridmet_single_pd = gridmet_single_pd[["date", var]]
        gridmet_single_pd = gridmet_single_pd.rename(columns={var: current_cell_id})
        #sentinel1_single_pd['date'] = sentinel1_single_pd['date'].astype('datetime64[ns]')
        gridmet_single_pd['date'] = pd.to_datetime(gridmet_single_pd['date']).dt.round("D").astype(str)
        gridmet_single_pd = gridmet_single_pd.drop_duplicates(subset=['date'], keep='first') # this will remove all the other values of the same day

        gridmet_all_df = pd.merge(gridmet_all_df, gridmet_single_pd, how='left', on="date")
      
    print(gridmet_all_df.shape)
    gridmet_all_df.to_csv(all_gridmet_file)
  
  
def prepare_testing_csv():
  """
  MOD model:
    input columns: [m, doy, ndsi]
    output column: [swe]
  Sentinel1 model:
    input columns: [m, doy, grd]
    output column: [swe]
  gridMET model:
    input columns: [m, doy, tmmn, tmmx, pr, vpd, eto, rmax, rmin, vs]
    output column: [swe]
  """
  all_ready_file = f"{github_dir}/data/ready_for_testing/all_ready_2.csv"
  if os.path.exists(all_ready_file):
    os.remove(all_ready_file)
  
  all_mod_file = f"{github_dir}/data/ready_for_testing/modis_all.csv"
  modis_all_pd = pd.read_csv(all_mod_file, header=0)
  all_sentinel1_file = f"{github_dir}/data/ready_for_testing/sentinel1_all.csv"
  sentinel1_all_pd = pd.read_csv(all_sentinel1_file, header=0)
  all_gridmet_eto_file = f"{github_dir}/data/ready_for_testing/gridmet_eto_all.csv"
  gridmet_eto_all_pd = pd.read_csv(all_gridmet_eto_file, header=0, index_col = 0)
  all_gridmet_pr_file = f"{github_dir}/data/ready_for_testing/gridmet_pr_all.csv"
  gridmet_pr_all_pd = pd.read_csv(all_gridmet_pr_file, header=0, index_col = 0)
  all_gridmet_rmax_file = f"{github_dir}/data/ready_for_testing/gridmet_rmax_all.csv"
  gridmet_rmax_all_pd = pd.read_csv(all_gridmet_rmax_file, header=0, index_col = 0)
  all_gridmet_rmin_file = f"{github_dir}/data/ready_for_testing/gridmet_rmin_all.csv"
  gridmet_rmin_all_pd = pd.read_csv(all_gridmet_rmin_file, header=0, index_col = 0)
  all_gridmet_tmmn_file = f"{github_dir}/data/ready_for_testing/gridmet_tmmn_all.csv"
  gridmet_tmmn_all_pd = pd.read_csv(all_gridmet_tmmn_file, header=0, index_col = 0)
  all_gridmet_tmmx_file = f"{github_dir}/data/ready_for_testing/gridmet_tmmx_all.csv"
  gridmet_tmmx_all_pd = pd.read_csv(all_gridmet_tmmx_file, header=0, index_col = 0)
  all_gridmet_vpd_file = f"{github_dir}/data/ready_for_testing/gridmet_vpd_all.csv"
  gridmet_vpd_all_pd = pd.read_csv(all_gridmet_vpd_file, header=0, index_col = 0)
  all_gridmet_vs_file = f"{github_dir}/data/ready_for_testing/gridmet_vs_all.csv"
  gridmet_vs_all_pd = pd.read_csv(all_gridmet_vs_file, header=0, index_col = 0)
  
  grid_terrain_file = f"{github_dir}/data/terrain/gridcells_terrainData.csv"
  grid_terrain_pd = pd.read_csv(grid_terrain_file, header=0, index_col = 1)
  
  print("modis_all_size: ", modis_all_pd.shape)
  print("sentinel1_all_size: ", sentinel1_all_pd.shape)
  print("gridmet_size: ", gridmet_rmax_all_pd.shape)
  print("gridmet_size: ", gridmet_eto_all_pd.shape)
  print("cell_size: ", len(submission_format_pd.index))
  print("station size: ", station_cell_mapper_pd.shape)
  print("training_feature_pd size: ", training_feature_pd.shape)
  print("testing_feature_pd size: ", testing_feature_pd.shape)
  
  all_training_pd = pd.DataFrame(columns=["cell_id", "year", "m", "doy", "ndsi", "grd", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "lat", "lon", "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe"])
  all_training_pd = all_training_pd.reset_index()
  #for current_cell_id in submission_format_pd.index:
  for index, row in gridmet_eto_all_pd.iterrows():
    dt = datetime.strptime(row[0], '%Y-%m-%d')
    month = dt.month
    year = dt.year
    doy = dt.timetuple().tm_yday
    print(f"Dealing {year} {doy}")
    if doy < 14 or doy > 21:
      continue
    count = 0
    for cell_id in row.index[1:]:
      count += 1
      print("processing ", count)
      eto = row[cell_id]
      swe = -1
      ndsi = modis_all_pd.loc[index, cell_id]
      grd = sentinel1_all_pd.loc[index, cell_id]
      #eto = gridmet_eto_all_pd.loc[index, cell_id]
      pr = gridmet_pr_all_pd.loc[index, cell_id]
      rmax = gridmet_rmax_all_pd.loc[index, cell_id]
      rmin = gridmet_rmin_all_pd.loc[index, cell_id]
      tmmn = gridmet_tmmn_all_pd.loc[index, cell_id]
      tmmx = gridmet_tmmx_all_pd.loc[index, cell_id]
      vpd = gridmet_vpd_all_pd.loc[index, cell_id]
      vs = gridmet_vs_all_pd.loc[index, cell_id]
      lat = grid_terrain_pd.loc[cell_id, "Longitude [deg]"]
      lon = grid_terrain_pd.loc[cell_id, "Latitude [deg]"]
      elevation = grid_terrain_pd.loc[cell_id, "Elevation [m]"]
      aspect = grid_terrain_pd.loc[cell_id, "Aspect [deg]"]
      curvature = grid_terrain_pd.loc[cell_id, "Curvature [ratio]"]
      slope = grid_terrain_pd.loc[cell_id, "Slope [deg]"]
      eastness = grid_terrain_pd.loc[cell_id, "Eastness [unitCirc.]"]
      northness = grid_terrain_pd.loc[cell_id, "Northness [unitCirc.]"]


      json_kv = {"cell_id": cell_id, "year":year, "m":month, "doy": doy, "ndsi":ndsi, "grd":grd, "eto":eto, "pr":pr, "rmax":rmax, "rmin":rmin, "tmmn":tmmn, "tmmx":tmmx, "vpd":vpd, "vs":vs, "lat":lat, "lon":lon, "elevation":elevation, "aspect":aspect, "curvature":curvature, "slope":slope, "eastness":eastness, "northness":northness, "swe":swe}
      #print(json_kv)
      all_training_pd = all_training_pd.append(json_kv, ignore_index = True)
      
      #break
  
  print(all_training_pd.shape)
  all_training_pd.to_csv(all_ready_file)
  
  
#exit() # done already

#integrate_modis()
#integrate_sentinel1()
#integrate_gridmet()
prepare_testing_csv()


  
  
  

