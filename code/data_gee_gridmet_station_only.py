import json
import pandas as pd
import ee
import os
import geopandas as gpd
import numpy as np
import concurrent.futures
import eeauth as e

# authenticate with Earth Engine
try:
    ee.Initialize(e.creds())
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# set up parameters
org_name = 'gridmet'
product_name = 'IDAHO_EPSCOR/GRIDMET'
start_date = '2018-10-01'
end_date = '2019-09-30'
var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']
homedir = os.path.expanduser('~')
github_dir = os.path.join(homedir, 'Documents', 'GitHub', 'SnowCast')
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"
station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)


# helper function to get data for a single cell
def get_cell_data(args):
    cell_id, longitude, latitude = args
    print(f'Running cell data for lat: {latitude}, long:{longitude}')
    try:
        # identify a 500 meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(longitude, latitude).buffer(1000)
        viirs = ee.ImageCollection(product_name).filterDate(start_date, end_date).filterBounds(poi).select(var_name)

        def poi_mean(img):
            reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=1000)
            mean = reducer.get(var_name)
            return img.set('date', img.date().format()).set(column_name, mean)

        poi_reduced_imgs = viirs.map(poi_mean)

        nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date', column_name]).values().get(0)

        # dont forget we need to call the callback method "getInfo" to retrieve the data
        df = pd.DataFrame(nested_list.getInfo(), columns=['date', column_name])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['cell_id'] = cell_id
        df['latitude'] = latitude
        df['longitude'] = longitude

        return df
    except Exception as e:
        print(e)
        return None


# iterate over variables and cells to retrieve data
for var_name in var_list:
    column_name = var_name
    dfolder = f"{homedir}/Documents/GitHub/SnowCast/data/sim_training/{org_name}/"
    if not os.path.exists(dfolder):
        os.makedirs(dfolder)

    all_cell_df = pd.DataFrame(columns=['date', column_name, 'cell_id', 'latitude', 'longitude'])
    cell_args = [(cell_id, longitude, latitude) for cell_id, longitude, latitude in
                 zip(station_cell_mapper_df['cell_id'], station_cell_mapper_df['lon'], station_cell_mapper_df['lat'])]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        cell_dfs = list(executor.map(get_cell_data, cell_args))

    for df in cell_dfs:
        if df is not None:
            df_list = [all_cell_df, df]
            all_cell_df = pd.concat(df_list)  # merge into big dataframe

    all_cell_df.to_csv(f"{dfolder}/{column_name}.csv")

