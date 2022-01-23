# Load dependencies
import geopandas as gpd
import json
import geojson
from pystac_client import Client
import planetary_computer
import xarray
import rioxarray
import xrspatial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Proj, transform

# user-defined paths for data-access
data_dir = '../../SnowCast_data/'
gridcells_file = data_dir+'grid_cells.geojson'
stations_file = data_dir+'ground_measures_metadata.csv'
gridcells_outfile = data_dir+'gridcells_terrainData.csv'
stations_outfile = data_dir+'station_terrainData.csv'

# setup client for handshaking and data-access
client = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    ignore_conformance=True,
)

# Load metadata
gridcellsGPD = gpd.read_file(gridcells_file)
gridcells = geojson.load(open(gridcells_file))
stations = pd.read_csv(stations_file)

# instantiate output panda dataframes
df_gridcells = df = pd.DataFrame(columns=("Longitude [deg]","Latitude [deg]",
                                          "Elevation [m]","Aspect [deg]",
                                          "Curvature [ratio]","Slope [deg]"))
df_station = pd.DataFrame(columns=("Longitude [deg]","Latitude [deg]",
                                   "Elevation [m]","Elevation_30 [m]","Elevation_1000 [m]",
                                   "Aspect_30 [deg]","Aspect_1000 [deg]",
                                   "Curvature_30 [ratio]","Curvature_1000 [ratio]",
                                   "Slope_30 [deg]","Slope_1000 [deg]"))

# Calculate gridcell characteristics using Copernicus DEM data
for idx,cell in enumerate(gridcells['features']):
    print(idx)
    search = client.search(
        collections=["cop-dem-glo-30"],
        intersects={"type":"Polygon", "coordinates":cell['geometry']['coordinates']},
    )
    items = list(search.get_items())   
    
    try:
        signed_asset = planetary_computer.sign(items[0].assets["data"])
        data = (
            xarray.open_rasterio(signed_asset.href)
            .squeeze()
            .drop("band")
            .coarsen({"y": 1, "x": 1})
            .mean()
        )
        cropped_data = data.rio.clip(gridcellsGPD['geometry'][idx:idx+1])
    except:
        signed_asset = planetary_computer.sign(items[1].assets["data"])
        data = (
            xarray.open_rasterio(signed_asset.href)
            .squeeze()
            .drop("band")
            .coarsen({"y": 1, "x": 1})
            .mean()
        )
        cropped_data = data.rio.clip(gridcellsGPD['geometry'][idx:idx+1])
    
    longitude = np.unique(np.ravel(cell['geometry']['coordinates'])[0::2]).mean()
    latitude = np.unique(np.ravel(cell['geometry']['coordinates'])[1::2]).mean()
    
    cropped_data = cropped_data.rio.reproject("EPSG:32612")
        
    mean_elev = cropped_data.mean().values
    print(mean_elev)
    
    aspect = xrspatial.aspect(cropped_data)
    aspect_xcomp = np.nansum(np.cos(aspect.values*(np.pi/180)))
    aspect_ycomp = np.nansum(np.sin(aspect.values*(np.pi/180)))
    mean_aspect = np.arctan2(aspect_ycomp,aspect_xcomp)*(180/np.pi)
    if mean_aspect < 0:
        mean_aspect = 360 + mean_aspect
    print(mean_aspect)
    
    # Positive curvature = upward convex
    curvature = xrspatial.curvature(cropped_data)
    mean_curvature = curvature.mean().values
    print(mean_curvature)
    
    slope = xrspatial.slope(cropped_data)
    mean_slope = slope.mean().values
    print(mean_slope)
    
    df_gridcells.loc[idx] = [longitude,latitude,
                             mean_elev,mean_aspect,
                             mean_curvature,mean_slope]
    
    if idx % 250 == 0:
        df_gridcells.set_index(gridcellsGPD['cell_id'][0:idx+1],inplace=True)
        df_gridcells.to_csv(gridcells_outfile)

# Save output data into csv format
df_gridcells.set_index(gridcellsGPD['cell_id'][0:idx+1],inplace=True)
df_gridcells.to_csv(gridcells_outfile)

# Calculate terrain characteristics of stations, and surrounding regions using COP 30
for idx,station in stations.iterrows():
    search = client.search(
        collections=["cop-dem-glo-30"],
        intersects={"type":"Point", "coordinates":[station['longitude'],station['latitude']]},
    )
    items = list(search.get_items())
    print(f"Returned {len(items)} items")
    
    try:
        signed_asset = planetary_computer.sign(items[0].assets["data"])
        data = (
            xarray.open_rasterio(signed_asset.href)
            .squeeze()
            .drop("band")
            .coarsen({"y": 1, "x": 1})
            .mean()
        )
        xdiff = np.abs(data.x-station['longitude'])
        ydiff = np.abs(data.y-station['latitude'])
        xdiff = np.where(xdiff == xdiff.min())[0][0]
        ydiff = np.where(ydiff == ydiff.min())[0][0]
        data = data[ydiff-33:ydiff+33,xdiff-33:xdiff+33].rio.reproject("EPSG:32612")
    except:
        signed_asset = planetary_computer.sign(items[1].assets["data"])
        data = (
            xarray.open_rasterio(signed_asset.href)
            .squeeze()
            .drop("band")
            .coarsen({"y": 1, "x": 1})
            .mean()
        )
        xdiff = np.abs(data.x-station['longitude'])
        ydiff = np.abs(data.y-station['latitude'])
        xdiff = np.where(xdiff == xdiff.min())[0][0]
        ydiff = np.where(ydiff == ydiff.min())[0][0]
        data = data[ydiff-33:ydiff+33,xdiff-33:xdiff+33].rio.reproject("EPSG:32612")
    
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:32612')
    new_x,new_y = transform(inProj,outProj,station['longitude'],station['latitude'])
    
    mean_elevation = data.mean().values
    elevation = data.sel(x=new_x,y=new_y,method='nearest')
    print(elevation.values)
    
    aspect = xrspatial.aspect(data)
    aspect_xcomp = np.nansum(np.cos(aspect.values*(np.pi/180)))
    aspect_ycomp = np.nansum(np.sin(aspect.values*(np.pi/180)))
    mean_aspect = np.arctan2(aspect_ycomp,aspect_xcomp)*(180/np.pi)
    if mean_aspect < 0:
        mean_aspect = 360 + mean_aspect
    print(mean_aspect)
    aspect = aspect.sel(x=new_x,y=new_y,method='nearest')
    print(aspect.values)
    
    # Positive curvature = upward convex
    curvature = xrspatial.curvature(data)
    mean_curvature = curvature.mean().values
    curvature = curvature.sel(x=new_x,y=new_y,method='nearest')
    print(curvature.values)
    
    slope = xrspatial.slope(data)
    mean_slope = slope.mean().values
    slope = slope.sel(x=new_x,y=new_y,method='nearest')
    print(slope.values)
    
    df_station.loc[idx] = [station['longitude'],station['latitude'],
                           station['elevation_m'],elevation.values,mean_elevation,
                           aspect.values,mean_aspect,
                           curvature.values,mean_curvature,
                           slope.values,mean_slope]
    
    if idx % 250 == 0:
        df_station.set_index(stations['station_id'][0:idx+1],inplace=True)
        df_station.to_csv(stations_outfile)

# Save output data into CSV format
df_station.set_index(stations['station_id'][0:idx+1],inplace=True)
df_station.to_csv(stations_outfile)


