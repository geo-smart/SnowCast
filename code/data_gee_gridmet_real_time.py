
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
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format.csv"
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
start_date = test_start_date
end_date = test_end_date

var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']

for var in var_list:

    var_name = var
    column_name = var

    dfolder = f"{homedir}/Documents/GitHub/SnowCast/data/sim_testing/{org_name}/"
    if not os.path.exists(dfolder):
        os.makedirs(dfolder)

    all_cell_df = pd.DataFrame(columns = ['date', column_name, 'cell_id', 'latitude', 'longitude'])

    for current_cell_id in submission_format_df.index:

        try:
          #print("collecting ", current_cell_id)
          single_csv_file = f"{dfolder}/{column_name}_{current_cell_id}.csv"

          if os.path.exists(single_csv_file):
              #os.remove(single_csv_file)
              #print("exists skipping..")
              continue

          longitude = all_cell_coords_pd['lon'][current_cell_id]
          latitude = all_cell_coords_pd['lat'][current_cell_id]

          # identify a 500 meter buffer around our Point Of Interest (POI)
          poi = ee.Geometry.Point(longitude, latitude).buffer(1000)
          viirs = ee.ImageCollection(product_name).filterDate(start_date, end_date).filterBounds(poi).select(var_name)

          def poi_mean(img):
              reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=1000)
              mean = reducer.get(var_name)
              return img.set('date', img.date().format()).set(column_name,mean)


          poi_reduced_imgs = viirs.map(poi_mean)

          nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date',column_name]).values().get(0)

          # dont forget we need to call the callback method "getInfo" to retrieve the data
          df = pd.DataFrame(nested_list.getInfo(), columns=['date',column_name])

          df['date'] = pd.to_datetime(df['date'])
          df = df.set_index('date')

          df['cell_id'] = current_cell_id
          df['latitude'] = latitude
          df['longitude'] = longitude
          df.to_csv(single_csv_file)

          df_list = [all_cell_df, df]
          all_cell_df = pd.concat(df_list) # merge into big dataframe

        except Exception as e:

          print(e)
          pass
    
    all_cell_df.to_csv(f"{dfolder}/{column_name}.csv")  



