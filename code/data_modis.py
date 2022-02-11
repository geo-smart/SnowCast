# Data Preparation MODIS

# print("Prepare MODIS into .csv")


# Load dependencies
import geopandas as gpd
import json
import geojson
from pystac_client import Client
import planetary_computer
import xarray
import rioxarray as rxr
import xrspatial
import rasterio
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Proj, transform
from shapely.geometry import mapping, box
import earthpy.spatial as es

import tempfile
import wget
import numpy as np
import matplotlib.pyplot as plt
import os

import datetime
import earthpy.plot as ep
from rasterio.plot import plotting_extent
import math

from azure.storage.blob import ContainerClient
import warnings
# MOD10A1 (Terra Snow Cover Daily L3 Global 500m)
# MOD10A2 (Terra Snow Cover 8-Day L3 Global 500m) -- not use
# MYD10A1 (Aqua Snow Cover Daily L3 Global 500m)
# MYD10A2 (Aqua Snow Cover 8-Day L3 Global 500m) -- not use

# 0-100=NDSI snow 200=missing data
# 201=no decision
# 211=night
# 237=inland water 239=ocean 250=cloud
# 254=detector saturated 255=fill

def lat_lon_to_modis_tile(lat,lon):
    """
    Get the modis tile indices (h,v) for a given lat/lon
    
    https://www.earthdatascience.org/tutorials/convert-modis-tile-to-lat-lon/
    """
    
    found_matching_tile = False
    i = 0
    while(not found_matching_tile):
        found_matching_tile = lat >= modis_tile_extents[i, 4] \
        and lat <= modis_tile_extents[i, 5] \
        and lon >= modis_tile_extents[i, 2] and lon <= modis_tile_extents[i, 3]
        i += 1
        
    v = int(modis_tile_extents[i-1, 0])
    h = int(modis_tile_extents[i-1, 1])
    
    return h,v

def lat_lon_to_modis_tile_distance(lat,lon):
    """
    Get the modis tile indices (h,v) for a given lat/lon 
    the least distance to the center of modis tile 
    
    https://www.earthdatascience.org/tutorials/convert-modis-tile-to-lat-lon/
    """
    
    # found_matching_tile = False
    Nrow = modis_tile_extents.shape[0]
    lat_dis = 99
    lon_dis = 99
    for i in range(0, Nrow-1):
        if lat <= modis_tile_extents[i,5] and lat >= modis_tile_extents[i,4] \
        and lon >= modis_tile_extents[i,2] and lon <= modis_tile_extents[i,3]:
            lat_cen = (modis_tile_extents[i,5]+modis_tile_extents[i,4])/2
            lon_cen = (modis_tile_extents[i,2]+modis_tile_extents[i,3])/2
            if(lat_dis >= abs(lat-lat_cen) and lon_dis >= abs(lon-lon_cen)):
                lat_dis = abs(lat-lat_cen)
                lon_dis = abs(lon-lon_cen)
                v = int(modis_tile_extents[i, 0])
                h = int(modis_tile_extents[i, 1])
            
    return h,v

def list_blobs_in_folder(container_name,folder_name):
    """
    List all blobs in a virtual folder in an Azure blob container
    """
    
    files = []
    generator = modis_container_client.list_blobs(name_starts_with=folder_name)
    for blob in generator:
        files.append(blob.name)
    return files
        
    
def list_hdf_blobs_in_folder(container_name,folder_name):
    """"
    List .hdf files in a folder
    """
    
    files = list_blobs_in_folder(container_name,folder_name)
    files = [fn for fn in files if fn.endswith('.hdf')]
    return files             


# user-defined paths for data-access
data_dir = '../../SnowCast/data/'
out_dir = data_dir + 'MODIS_SCA/'
gridcells_file = data_dir+'station_gridcell/grid_cells.geojson'


# Load metadata
gridcellsGPD = gpd.read_file(gridcells_file)
gridcells = geojson.load(open(gridcells_file))
use_proj = gridcellsGPD.crs


modis_account_name = 'modissa'
modis_container_name = 'modis-006'
modis_account_url = 'https://' + modis_account_name + '.blob.core.windows.net/'
modis_blob_root = modis_account_url + modis_container_name + '/'

# This file is provided by NASA; it indicates the lat/lon extents of each
# MODIS tile.
#
# The file originally comes from:
#
# https://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt
modis_tile_extents_url = modis_blob_root + 'sn_bound_10deg.txt'

temp_dir = os.path.join(out_dir,'modis_images')
os.makedirs(temp_dir,exist_ok=True)
fn = os.path.join(temp_dir,modis_tile_extents_url.split('/')[-1])
wget.download(modis_tile_extents_url, fn)

# Load this file into a table, where each row is (v,h,lonmin,lonmax,latmin,latmax)
modis_tile_extents = np.genfromtxt(fn,
                     skip_header = 7, 
                     skip_footer = 3)

modis_container_client = ContainerClient(account_url=modis_account_url, 
                                         container_name=modis_container_name,
                                         credential=None)

# Files are stored according to:
#
# http://modissa.blob.core.windows.net/modis-006/[product]/[htile]/[vtile]/[year][day]/filename

# This is the MODIS surface reflectance product
product_list = ['MOD10A1','MYD10A1']

# Search daily MODIS data during snow season
start = datetime.datetime.strptime("20130101", "%Y%m%d")
end = datetime.datetime.strptime("20211231", "%Y%m%d")

# end = datetime.datetime.strptime("20130110", "%Y%m%d")

date_search = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
# data = {"Date": date_search, "Month": [m.strftime("%m") for m in date_search]}
df_date_search = pd.DataFrame(data = {"Date": date_search, "Month": [m.strftime("%m") for m in date_search]})
df_date_search = df_date_search[df_date_search["Month"].isin(["01","02","03","04","05","06","10","11","12"])]

    
for product in product_list:
    # instantiate output panda dataframes
    gridcells_outfile = out_dir+'gridcells_' + product + '.csv'
    column_names = [c.strftime("%Y%m%d") for c in df_date_search["Date"]]
    column_names.insert(0,"cell_id")
    df_gridcells = pd.DataFrame(columns = column_names)
    
    for idx,cell in enumerate(gridcells['features']):
        gridcell_boundary = gridcellsGPD['geometry'][idx:idx+1]
        df_gridcells.loc[idx,"cell_id"] = gridcellsGPD['cell_id'][idx]

        # get the h and v value based on centrol lat and lon of each gridcell
        h,v = lat_lon_to_modis_tile_distance(gridcell_boundary.centroid.y.values,gridcell_boundary.centroid.x.values)

        # find all available MOD10A1 and MYD10A1 for each grid for the entire modeling period
        # this takes quite a long time 
        # can be done in parallele 
        index = 0
        for idate in df_date_search["Date"]:
            # date_search = datetime.datetime.strptime(idate,"%Y%m%d")
            print(idate)
            
            year = idate.strftime("%Y")
            doy = idate.strftime('%j')
            doy_search = str(year) + doy

            folder = product + '/' + '{:0>2d}/{:0>2d}'.format(h,v) + '/' + doy_search
            # Find all HDF files from this tile on this day
            filenames = list_hdf_blobs_in_folder(modis_container_name,folder)
            # print(filenames)
            # print('Found {} matching file(s):'.format(len(filenames)))
            # for fn in filenames:
            #     print(fn)
            if len(filenames) > 0:

                # Work with the first returned URL; Generally, only one image will be available
                blob_name = filenames[0]
                # print(blob_name)

                # Get URL to download to a temporary file
                url = modis_blob_root + blob_name
                filename = os.path.join(temp_dir,blob_name.replace('/','_'))
                # if file not exist, then download the file
                if not os.path.isfile(filename):
                    wget.download(url,filename)


                modis_sca = rxr.open_rasterio(filename, variable=['NDSI_Snow_Cover'])
                # reproject modis sca
                #By default, pixel values are read raw or interpolated using a nearest neighbor algorithm from the band cache
                modis_sca_repo = modis_sca.rio.reproject(gridcellsGPD.crs) 
                modis_sca_clip = modis_sca_repo.rio.clip(gridcell_boundary, all_touched = True)
                # print(modis_sca_clip)
                sca = modis_sca_clip.where(modis_sca_clip.isin(range(0,100)), drop = True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    sca = np.nanmean(sca.to_array().values)
                # print(sca)
                if not math.isnan(sca):
                    sca_mean = sca
                else:
                    sca_mean = -999
                
                print(sca_mean)
   
                df_gridcells.iloc[idx,index+1] = sca_mean
                df_gridcells.to_csv(gridcells_outfile)
                
            index = index + 1
           
                    
        
       
   
        
