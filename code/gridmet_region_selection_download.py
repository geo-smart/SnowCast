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
import multiprocessing

# Initialize GEE
# try:
#     ee.Initialize()
# except Exception as e:
#     # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
#     ee.Authenticate()
#     ee.Initialize()

service_account = 'eartheginegcloud@earthengine58.iam.gserviceaccount.com'
creds = ee.ServiceAccountCredentials(
    service_account, '/home/chetana/bhargavi-creds.json')

ee.Initialize(creds)

# Read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)

# Read grid cell
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"
submission_format_df = pd.read_csv(
    submission_format_file, header=0, index_col=0)
all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
all_cell_coords_pd = pd.read_csv(all_cell_coords_file, header=0, index_col=0)

print(submission_format_df.shape)

# Set the variables
org_name = 'gridmet'
product_name = 'IDAHO_EPSCOR/GRIDMET'
start_date = findLastStopDate(
    f"{github_dir}/data/sim_testing/{org_name}/", "%Y-%m-%d %H:%M:%S")
end_date = test_end_date
var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']

# Create a list of cell IDs and their coordinates
cell_coords = [(cell_id, all_cell_coords_pd['lon'][cell_id],
                all_cell_coords_pd['lat'][cell_id]) for cell_id in submission_format_df.index]

# change this
start_date = '2022-01-01'
end_date = '2022-04-04'
#

# Define a function to retrieve data for a batch of cells


def get_batch_data(batch):
    cell_data_list = []
    for cell_id, longitude, latitude in batch:
        try:
            print(f"=> Collected GridMet data for {cell_id}")
            poi = ee.Geometry.Point(longitude, latitude).buffer(1000)
            viirs = ee.ImageCollection(product_name).filterDate(
                start_date, end_date).filterBounds(poi).select(var_list)

            def poi_mean(img):
                reducer = img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=poi, scale=1000)
                img = img.set('date', img.date().format())
                for var in var_list:
                    column_name = var
                    mean = reducer.get(column_name)
                    img = img.set(column_name, mean)
                return img

            poi_reduced_imgs = viirs.map(poi_mean)

            nested_list = poi_reduced_imgs.reduceColumns(
                ee.Reducer.toList(var_list), ['date']).get('list')

            df = pd.DataFrame.from_records(nested_list.getInfo(), index='date')
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            df['cell_id'] = cell_id
            df['latitude'] = latitude
            df['longitude'] = longitude

            cell_data_list.append(df)
        except Exception as e:
            print(f"=> Error collecting data for cell {cell_id}")
            print(traceback.format_exc())

    return pd.concat(cell_data_list)


# Define a function to collect data for all cells using multiple processes
def collect_all_data(cell_coords, num_processes=4):
    with multiprocessing.Pool(num_processes) as p:
        results = p.map(get_batch_data, np.array_split(
            cell_coords, num_processes))
        return pd.concat(results)


num_processes = 4
collect_all_data(cell_coords, num_processes=4)

# try:
#     print("Collecting data for all cells...")
#     cell_data = collect_all_data(cell_coords, num_processes=num_processes)
#     print("Data collection complete!")
# except Exception as e:
#     print(f"Error collecting data for all cells: {e}")
#     print(traceback.format_exc())
# output_dir = f"{github_dir}/data/sim_testing/{org_name}/"
# output_filename = f"{output_dir}/{start_date.strftime('%Y-%m-%d_%H:%M:%S')}{end_date.strftime('%Y-%m-%d%H:%M:%S')}.csv"

# try:
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         cell_data.to_csv(output_filename)
#         print(f"Data saved to {output_filename}")
# except Exception as e:
#     print(f"Error saving data to {output_filename}: {e}")
#     print(traceback.format_exc())

