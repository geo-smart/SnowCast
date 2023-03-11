from datetime import date
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
import datetime

today = date.today()

# dd/mm/YY
d1 = today.strftime("%Y-%m-%d")
print("today date =", d1)

train_start_date = ""
train_end_date = ""

test_start_date = "2022-01-01"
test_end_date = d1

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"


def calculateDistance(lat1, lon1, lat2, lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    return math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)

def create_cell_location_csv():
  # read grid cell
  gridcells_file = f"{github_dir}/data/snowcast_provided/grid_cells_eval.geojson"
  all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
  if os.path.exists(all_cell_coords_file):
    os.remove(all_cell_coords_file)

  grid_coords_df = pd.DataFrame(columns=["cell_id", "lat", "lon"])
  #print(grid_coords_df.head())
  gridcells = geojson.load(open(gridcells_file))
  for idx,cell in enumerate(gridcells['features']):
    
    current_cell_id = cell['properties']['cell_id']
    cell_lon = np.unique(np.ravel(cell['geometry']['coordinates'])[0::2]).mean()
    cell_lat = np.unique(np.ravel(cell['geometry']['coordinates'])[1::2]).mean()
    grid_coords_df.loc[len(grid_coords_df.index)] = [current_cell_id, cell_lat, cell_lon]
    
  #grid_coords_np = grid_coords_df.to_numpy()
  #print(grid_coords_np.shape)
  grid_coords_df.to_csv(all_cell_coords_file, index=False)
  #np.savetxt(all_cell_coords_file, grid_coords_np[:, 1:], delimiter=",")
  #print(grid_coords_np.shape)
  
def get_latest_date_from_an_array(arr, date_format):
  return max(arr, key=lambda x: datetime.datetime.strptime(x, date_format))
  
  
def findLastStopDate(target_testing_dir, data_format):
  date_list = []
  for filename in os.listdir(target_testing_dir):
    
    f = os.path.join(target_testing_dir, filename)
    # checking if it is a file
    if os.path.isfile(f) and ".csv" in f:
        pdf = pd.read_csv(f,header=0, index_col=0)
        #print(pdf)
        date_list = np.concatenate((date_list, pdf.index.unique()))
        
  latest_date = get_latest_date_from_an_array(date_list, data_format)
  #print(latest_date)
  date_time_obj = datetime.datetime.strptime(latest_date, data_format)
  return date_time_obj.strftime("%Y-%m-%d")

#create_cell_location_csv()
findLastStopDate(f"/home/chetana/Documents/GitHub/SnowCast/data/sim_training/gridmet/", "%Y-%m-%d %H:%M:%S")
#findLastStopDate(f"{github_dir}/data/sat_testing/sentinel1/", "%Y-%m-%d %H:%M:%S")
#findLastStopDate(f"{github_dir}/data/sat_testing/modis/", "%Y-%m-%d")



      

