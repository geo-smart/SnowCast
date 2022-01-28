# Data preparation for Sentinel 1

import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import panel as pn
import intake
import numpy as np
import os
import pandas as pd
import rasterio
import rioxarray
import s3fs 
import xarray as xr
hv.extension('bokeh')

print("prepare Sentinel 1 into .csv")

# GDAL environment variables to efficiently read remote data
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR' #This is KEY! otherwise we send a bunch of HTTP GET requests to test for common sidecar metadata
os.environ['AWS_NO_SIGN_REQUEST']='YES' #Since this is a public bucket, we don't need authentication
os.environ['GDAL_MAX_RAW_BLOCK_CACHE_SIZE']='200000000'  #200MB: Default is 10 MB limit in the GeoTIFF driver for range request merging.

# Data is stored in a public S3 Bucket
url = 's3://sentinel-s1-rtc-indigo/tiles/RTC/1/IW/12/S/YJ/2016/S1B_20161121_12SYJ_ASC/Gamma0_VV.tif'

# These Cloud-Optimized-Geotiff (COG) files have 'overviews', low-resolution copies for quick visualization
da = rioxarray.open_rasterio(url, overview_level=3).squeeze('band')
da.hvplot.image(clim=(0,0.4), cmap='gray', 
                x='x', y='y', 
                aspect='equal', frame_width=400,
                title='S1B_20161121_12SYJ_ASC',
                rasterize=True # send rendered image to browser, rather than full array
               )

# Load a time series we created a VRT with GDAL to facilitate this step
da3 = rioxarray.open_rasterio(vrtName, overview_level=3, chunks='auto')

# Need to add time coordinates to this data
datetimes = [pd.to_datetime(x[55:63]) for x in keys]
    
# add new coordinate to existing dimension 
da = da3.assign_coords(time=('band', datetimes))
# make 'time' active coordinate instead of integer band
da = da.swap_dims({'band':'time'})
# Name the dataset (helpful for hvplot calls later on)
da.name = 'Gamma0VV'

#use a small bounding box over grand mesa (UTM coordinates)
xmin,xmax,ymin,ymax = [739186, 742748, 4.325443e+06, 4.327356e+06]
daT = da.sel(x=slice(xmin, xmax), 
             y=slice(ymax, ymin))

# NOTE: this can take a while on slow internet connections, we're reading over 100 images!
all_points = daT.where(daT!=0).hvplot.scatter('time', groupby=[], dynspread=True, datashade=True) 
mean_trend = daT.where(daT!=0, drop=True).mean(dim=['x','y']).hvplot.line(title='North Grand Mesa', color='red')

path = '/tmp/tutorial-data/sar/sentinel1/S1AA_20201030T131820_20201111T131820_VVP012_INT80_G_ueF_EBD2/S1AA_20201030T131820_20201111T131820_VVP012_INT80_G_ueF_EBD2_unw_phase.tif'
da = rioxarray.open_rasterio(path, masked=True).squeeze('band')



