
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
from datetime import date
from snowcast_utils import *
import traceback


# exit() # uncomment to download new files

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate() # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Initialize()

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
# read grid cell
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"
submission_format_df = pd.read_csv(submission_format_file, header=0, index_col=0)
all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
all_cell_coords_pd = pd.read_csv(all_cell_coords_file, header=0, index_col=0)

print(submission_format_df.shape)

#org_name = 'modis'
#product_name = f'MODIS/006/MOD10A1'
#var_name = 'NDSI'
#column_name = 'mod10a1_ndsi'

org_name = 'gridmet'
product_name = 'IDAHO_EPSCOR/GRIDMET'
#start_date = "2022-04-20"#test_start_date
start_date = findLastStopDate(f"{github_dir}/data/sim_testing/{org_name}/", "%Y-%m-%d %H:%M:%S")
end_date = test_end_date
#start_date = "2022-04-06"
#end_date = "2022-04-18"

var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']


dfolder = f"{homedir}/Documents/GitHub/SnowCast/data/sim_testing/{org_name}/"
if not os.path.exists(dfolder):
  os.makedirs(dfolder)
  
column_list = ['date', 'cell_id', 'latitude', 'longitude']
column_list.extend(var_list)
reduced_column_list = ['date']
reduced_column_list.extend(var_list)

all_cell_df = pd.DataFrame(columns = column_list)

count = 0

for current_cell_id in submission_format_df.index:

  try:
    count+=1
    print(f"=> Collected GridMet data for {count} cells")
    print("collecting ", current_cell_id)
    #single_csv_file = f"{dfolder}/{column_name}_{current_cell_id}.csv"

    #if os.path.exists(single_csv_file):
    #  os.remove(single_csv_file)
    #  print("exists skipping..")
    #  continue

    longitude = all_cell_coords_pd['lon'][current_cell_id]
    latitude = all_cell_coords_pd['lat'][current_cell_id]

    # identify a 500 meter buffer around our Point Of Interest (POI)
    poi = ee.Geometry.Point(longitude, latitude).buffer(1000)
    viirs = ee.ImageCollection(product_name).filterDate(start_date, end_date).filterBounds(poi).select(var_list)

    def poi_mean(img):
      reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=1000)
      img = img.set('date', img.date().format());
      for var in var_list:
        column_name = var
        mean = reducer.get(column_name)
        img = img.set(column_name,mean)
      return img


    poi_reduced_imgs = viirs.map(poi_mean)

    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(9), reduced_column_list).values().get(0)

    # dont forget we need to call the callback method "getInfo" to retrieve the data
    df = pd.DataFrame(nested_list.getInfo(), columns=reduced_column_list)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df['cell_id'] = current_cell_id
    df['latitude'] = latitude
    df['longitude'] = longitude
    #df.to_csv(single_csv_file)

    #print(df.head())
    
    df_list = [all_cell_df, df]
    all_cell_df = pd.concat(df_list) # merge into big dataframe
    
    #if count % 4 == 0:

  except Exception as e:
    print(traceback.format_exc())
    print("Failed: ", e)
    pass

all_cell_df.to_csv(f"{dfolder}/all_vars_{start_date}_{end_date}.csv")  



