

# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernal

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
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"
station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)

start_date = '2022-02-20'
end_date = '2022-02-27'

org_name = 'sentinel1'
product_name = 'COPERNICUS/S1_GRD'
var_name = 'VV'
column_name = 's1_grd_vv'

final_csv_file = f"{homedir}/Documents/GitHub/SnowCast/data/sat_testing/{org_name}/{column_name}_{start_date}_{end_date}.csv"
print(f"Results will be saved to {final_csv_file}")


if os.path.exists(final_csv_file):
    print("exists skipping..")
    exit()


all_cell_df = pd.DataFrame(columns = ['date', column_name, 'cell_id', 'latitude', 'longitude'])

for ind in station_cell_mapper_df.index:
  
    try:
  	
      current_cell_id = station_cell_mapper_df['cell_id'][ind]
      print("collecting ", current_cell_id)
      
      longitude = station_cell_mapper_df['lon'][ind]
      latitude = station_cell_mapper_df['lat'][ind]

      # identify a 500 meter buffer around our Point Of Interest (POI)
      poi = ee.Geometry.Point(longitude, latitude).buffer(1)

      viirs = ee.ImageCollection(product_name) \
          	.filterDate(start_date, end_date) \
            .filterBounds(poi) \
          	.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
      		.select('VV')
      
      def poi_mean(img):
          reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi)
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

      df_list = [all_cell_df, df]
      all_cell_df = pd.concat(df_list) # merge into big dataframe
      
    except Exception as e:
      
      print(e)
      pass
    
all_cell_df.to_csv(final_csv_file)  



