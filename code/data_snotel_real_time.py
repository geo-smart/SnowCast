

# Write first python in Geoweaver
import pandas as pd
import os
import urllib.request, urllib.error, urllib.parse
import sys
import geojson
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
from snowcast_utils import *
import traceback

from datetime import datetime
from metloom.pointdata import SnotelPointData

print(sys.path)

try:
  from BeautifulSoup import BeautifulSoup
except ImportError:
  from bs4 import BeautifulSoup

print("testing...")
def get_nearest_stations_for_all_grids():
  
  print("Start to get nearest stations..")
  
  cache_file = f"{github_dir}/data/snowcast_provided/nearest_stations_grid_eval.pkl"
  
  if os.path.exists(cache_file):
    with open(cache_file, "rb") as fp:
      return pickle.load(fp)
  
  filepath = f"{github_dir}/data/snowcast_provided/grid_cells_eval.geojson"
  # Read in grid cells
  with open(filepath) as f:
    grid_cells = geojson.load(f)
  
  # Retrieve and store cell IDs and their corresponding coordinates
  cell_ID = []
  ID_to_coord = {}

  for i in range(len(grid_cells['features'])):
    id = grid_cells['features'][i]['properties']['cell_id']
    coords = grid_cells['features'][i]['geometry']['coordinates']
    if id is not None and coords is not None: 
      cell_ID.append(id)
      ID_to_coord[id] = coords

  nohrsc_url_format_string = "https://www.nohrsc.noaa.gov/nearest/index.html?city={lat}%2C{lon}&county=&l=5&u=e&y={year}&m={month}&d={day}"
  
  query_urls = []
  
  stations = []
  
  current_date = datetime.now()

  # Retrieve nearest stations for each grid cell
  for i in range(int(len(cell_ID))):
    for j in range(1):
      latitude = round(ID_to_coord.get(cell_ID[i])[0][j][1], 2)
      longitude = round(ID_to_coord.get(cell_ID[i])[0][j][0], 2)

      response = urllib.request.urlopen(nohrsc_url_format_string.format(lat = latitude, lon = longitude, year = current_date.year, month = current_date.month, day = current_date.day))
      webContent = response.read().decode('UTF-8')
      parsed_html = BeautifulSoup(webContent)

      snow_water_eq_table = parsed_html.find_all('table')[7]
      nearest_station = snow_water_eq_table.find_all('td')[0].text

      stations.append(nearest_station)
  
  # save stations somewhere for cache
  
  print("Nearest station list is completed. Saving to cache..")
  
  with open(f"{github_dir}/data/snowcast_provided/nearest_stations_grid_eval.pkl", "wb") as fp:
    pickle.dump(stations, fp)
          
  return stations

def get_snotel_time_series(start_date = '2016-1-1', end_date = '2022-7-12'):
  
  print(f"Retrieving data from {start_date} to {end_date}")
  # read the grid geometry file
  homedir = os.path.expanduser('~')
  print(homedir)
  # read grid cell
  github_dir = f"{homedir}/Documents/GitHub/SnowCast"
  # read grid cell
  #submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"
  #all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
  #all_cell_coords_pd = pd.read_csv(all_cell_coords_file, header=0, index_col=0)

  stations = get_nearest_stations_for_all_grids()

  start_date = datetime.strptime(start_date, "%Y-%m-%d")
  end_date = datetime.strptime(end_date, "%Y-%m-%d")
  # NOTE: this url only allows user to query one year's worth of data
  nohrsc_time_series_url = 'https://www.nohrsc.noaa.gov/interactive/html/graph.html?station={station}&w=600&h=400&o=a&uc=0&by={start_year}&bm={start_month}&bd={start_day}&bh=0&ey={end_year}&em={end_month}&ed={end_day}&eh=23&data=1&units=0&region=us'

  dates = []
  snow_water_eq = []
  # Keep track of cell IDs for each time series
  new_cell_IDs = []

  # Retrieve time series snow water equivalent data for each grid cell's nearest station
  for i in range(len(stations)):

    if stations[i] != '\nNo observations within 62 miles\n':

      for j in range(end_date.year - start_date.year + 1):

        response = urllib.request.urlopen(nohrsc_time_series_url.format(station = stations[i], start_year = start_date.year + j, start_month = start_date.month, start_day = start_date.day, end_year = end_date.year, end_month = end_date.month, end_day = end_date.day))
        webContent = response.read().decode('UTF-8')
        parsed_html = BeautifulSoup(webContent)

        data_table = parsed_html.find_all('tbody')[3].find_all('tr')

        for k in range(len(data_table)):
          new_cell_IDs.append(cell_ID[i])
          dates.append(data_table[k].find_all('td')[0].text)
          snow_water_eq.append(data_table[k].find_all('td')[2].text)
    else:
      new_cell_IDs.append(cell_ID[i])
      dates.append('N/A')
      snow_water_eq.append('N/A')

  df = pd.DataFrame({'cell_IDs': new_cell_IDs, 'date_times': dates, 'swe': snow_water_eq})
  df.to_csv(f"{github_dir}/data/snotel/SNOTEL_time_series_all_grid_eval.csv")
  print("All SNOTEL time series are saved.")

get_nearest_stations_for_all_grids()
#get_snotel_time_series("2022-06-01", "2022-06-02")

