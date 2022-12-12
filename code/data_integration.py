# Integrate all the datasets into one training dataset
import json
import pandas as pd
import ee
import seaborn as sns
import matplotlib.pyplot as plt
from math import radians
from sklearn import neighbors as sk
import os
import geopandas as gpd
import geojson
import numpy as np
import os.path
from datetime import datetime,timedelta

print("integrating datasets into one dataset")
# pd.set_option('display.max_columns', None)

# read the grid geometry file
homedir = os.path.expanduser('~')
print(homedir)
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
gridcells_file = f"{github_dir}/data/snowcast_provided/grid_cells.geojson"
model_dir = f"{github_dir}/model/"
training_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_train_features.csv"
testing_feature_file = f"{github_dir}/data/snowcast_provided/ground_measures_test_features.csv"
train_labels_file = f"{github_dir}/data/snowcast_provided/train_labels.csv"
ground_measure_metadata_file = f"{github_dir}/data/snowcast_provided/ground_measures_metadata.csv"
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"

# example_mod_file = f"{github_dir}/data/modis/mod10a1_ndsi_f191fe19-0e81-4bc9-9980-29738a05a49b.csv"


training_feature_pd = pd.read_csv(training_feature_file, header=0, index_col=0)
testing_feature_pd = pd.read_csv(testing_feature_file, header=0, index_col=0)
train_labels_pd = pd.read_csv(train_labels_file, header=0, index_col=0)
print(train_labels_pd.head())
# if "2ca6a37f-67f5-4905-864b-ddf98d956ebb" in train_labels_pd.index and "2013-01-02" in train_labels_pd.columns:
#   print("Check one value: ", train_labels_pd.loc["2ca6a37f-67f5-4905-864b-ddf98d956ebb"]["2013-01-02"])
# else:
#   print("Key not existed")

station_cell_mapper_pd = pd.read_csv(station_cell_mapper_file, header=0, index_col=0)


# print(station_cell_mapper_pd.head())

# example_mod_pd = pd.read_csv(example_mod_file, header=0, index_col=0)
# print(example_mod_pd.shape)


def getDateStr(x):
    return x.split(" ")[0]


def integrate_modis():
    """
  Integrate all MODIS data into mod_all.csv
  """
    all_mod_file = f"{github_dir}/data/ready_for_training/modis_all.csv"
    if os.path.isfile(all_mod_file):
        return
    dates = pd.date_range(start='1/1/2013', end='12/31/2021', freq='D').astype(str)
    mod_all_df = pd.DataFrame(columns=["date"])
    mod_all_df['date'] = dates

    # print(mod_all_df.head())
    for ind in station_cell_mapper_pd.index:
        current_cell_id = station_cell_mapper_pd["cell_id"][ind]
        print(current_cell_id)
        mod_single_file = f"{github_dir}/data/sat_training/modis/mod10a1_ndsi_{current_cell_id}.csv"
        if os.path.isfile(mod_single_file):
            mod_single_pd = pd.read_csv(mod_single_file, header=0)
            mod_single_pd = mod_single_pd[["date", "mod10a1_ndsi"]]
            mod_single_pd = mod_single_pd.rename(columns={"mod10a1_ndsi": current_cell_id})
            mod_single_pd['date'] = pd.to_datetime(mod_single_pd['date']).astype(str)
            print(mod_all_df.shape)
            mod_all_df = pd.merge(mod_all_df, mod_single_pd, how='left', on="date")
    mod_all_df.to_csv(all_mod_file)


def integrate_sentinel1():
    """
  Integrate all Sentinel 1 data into sentinel1_all.csv
  """
    all_sentinel1_file = f"{github_dir}/data/ready_for_training/sentinel1_all.csv"
    if os.path.isfile(all_sentinel1_file):
        return
    dates = pd.date_range(start='1/1/2013', end='12/31/2021', freq='D').astype(str)
    sentinel1_all_df = pd.DataFrame(columns=["date"])
    sentinel1_all_df['date'] = dates
    # print(mod_all_df.head())

    for ind in station_cell_mapper_pd.index:
        current_cell_id = station_cell_mapper_pd["cell_id"][ind]
        print(current_cell_id)
        sentinel1_single_file = f"{github_dir}/data/sat_training/sentinel1/s1_grd_vv_{current_cell_id}.csv"
        if os.path.isfile(sentinel1_single_file) and current_cell_id not in sentinel1_all_df:
            sentinel1_single_pd = pd.read_csv(sentinel1_single_file, header=0)
            sentinel1_single_pd = sentinel1_single_pd[["date", "s1_grd_vv"]]
            sentinel1_single_pd = sentinel1_single_pd.rename(columns={"s1_grd_vv": current_cell_id})
            # sentinel1_single_pd['date'] = sentinel1_single_pd['date'].astype('datetime64[ns]')
            sentinel1_single_pd['date'] = pd.to_datetime(sentinel1_single_pd['date']).dt.round("D").astype(str)
            print("sentinel1_single_pd: ", sentinel1_single_pd.head())
            print("sentinel1_single_pd check value: ", sentinel1_single_pd[sentinel1_single_pd["date"] == "2015-04-01"])
            sentinel1_single_pd = sentinel1_single_pd.drop_duplicates(subset=['date'],
                                                                      keep='first')  # this will remove all the other values of the same day

            sentinel1_all_df = pd.merge(sentinel1_all_df, sentinel1_single_pd, how='left', on="date")
            print("sentinel1_all_df check value: ", sentinel1_all_df[sentinel1_all_df["date"] == "2015-04-01"])
            print("sentinel1_all_df: ", sentinel1_all_df.shape)

    print(sentinel1_all_df.shape)
    sentinel1_all_df.to_csv(all_sentinel1_file)


def integrate_gridmet():
    """
  Integrate all gridMET data into gridmet_all.csv
  """

    dates = pd.date_range(start='10/1/2018', end='09/30/2019', freq='D').astype(str)

    # print(mod_all_df.head())
    var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']

    for var in var_list:
        gridmet_all_df = pd.DataFrame(columns=["date"])
        gridmet_all_df['date'] = dates
        all_gridmet_file = f"{github_dir}/data/ready_for_training/gridmet_{var}_all.csv"
        if os.path.isfile(all_gridmet_file):
            return
        for ind in station_cell_mapper_pd.index:
            current_cell_id = station_cell_mapper_pd["cell_id"][ind]
            print(current_cell_id)
            gridmet_single_file = f"{github_dir}/data/sim_training/gridmet/{var}_{current_cell_id}.csv"
            if os.path.isfile(gridmet_single_file) and current_cell_id not in gridmet_all_df:
                gridmet_single_pd = pd.read_csv(gridmet_single_file, header=0)
                gridmet_single_pd = gridmet_single_pd[["date", var]]
                gridmet_single_pd = gridmet_single_pd.rename(columns={var: current_cell_id})
                # sentinel1_single_pd['date'] = sentinel1_single_pd['date'].astype('datetime64[ns]')
                gridmet_single_pd['date'] = pd.to_datetime(gridmet_single_pd['date']).dt.round("D").astype(str)
                print("gridmet_single_pd: ", gridmet_single_pd.head())
                print("gridmet_single_pd check value: ", gridmet_single_pd[gridmet_single_pd["date"] == "2015-04-01"])
                gridmet_single_pd = gridmet_single_pd.drop_duplicates(subset=['date'],
                                                                      keep='first')  # this will remove all the other values of the same day

                gridmet_all_df = pd.merge(gridmet_all_df, gridmet_single_pd, how='left', on="date")
                print("gridmet_all_df check value: ", gridmet_all_df[gridmet_all_df["date"] == "2015-04-01"])
                print("gridmet_all_df: ", gridmet_all_df.shape)

        print(gridmet_all_df.shape)
        gridmet_all_df.to_csv(all_gridmet_file)


def prepare_training_csv():
    """
  MOD model:
    input columns: [m, doy, ndsi]
    output column: [swe]
  Sentinel1 model:
    input columns: [m, doy, grd]
    output column: [swe]
  gridMET model:
    input columns: [m, doy, tmmn, tmmx, pr, vpd, eto, rmax, rmin, vs]
    output column: [swe]
  """
    all_ready_file = f"{github_dir}/data/ready_for_training/all_ready.csv"
    if os.path.isfile(all_ready_file):
        return

    all_mod_file = f"{github_dir}/data/ready_for_training/modis_all.csv"
    modis_all_pd = pd.read_csv(all_mod_file, header=0)
    all_sentinel1_file = f"{github_dir}/data/ready_for_training/sentinel1_all.csv"
    sentinel1_all_pd = pd.read_csv(all_sentinel1_file, header=0)
    all_gridmet_eto_file = f"{github_dir}/data/ready_for_training/gridmet_eto_all.csv"
    gridmet_eto_all_pd = pd.read_csv(all_gridmet_eto_file, header=0, index_col=0)
    all_gridmet_pr_file = f"{github_dir}/data/ready_for_training/gridmet_pr_all.csv"
    gridmet_pr_all_pd = pd.read_csv(all_gridmet_pr_file, header=0, index_col=0)
    all_gridmet_rmax_file = f"{github_dir}/data/ready_for_training/gridmet_rmax_all.csv"
    gridmet_rmax_all_pd = pd.read_csv(all_gridmet_rmax_file, header=0, index_col=0)
    all_gridmet_rmin_file = f"{github_dir}/data/ready_for_training/gridmet_rmin_all.csv"
    gridmet_rmin_all_pd = pd.read_csv(all_gridmet_rmin_file, header=0, index_col=0)
    all_gridmet_tmmn_file = f"{github_dir}/data/ready_for_training/gridmet_tmmn_all.csv"
    gridmet_tmmn_all_pd = pd.read_csv(all_gridmet_tmmn_file, header=0, index_col=0)
    all_gridmet_tmmx_file = f"{github_dir}/data/ready_for_training/gridmet_tmmx_all.csv"
    gridmet_tmmx_all_pd = pd.read_csv(all_gridmet_tmmx_file, header=0, index_col=0)
    all_gridmet_vpd_file = f"{github_dir}/data/ready_for_training/gridmet_vpd_all.csv"
    gridmet_vpd_all_pd = pd.read_csv(all_gridmet_vpd_file, header=0, index_col=0)
    all_gridmet_vs_file = f"{github_dir}/data/ready_for_training/gridmet_vs_all.csv"
    gridmet_vs_all_pd = pd.read_csv(all_gridmet_vs_file, header=0, index_col=0)

    grid_terrain_file = f"{github_dir}/data/terrain/gridcells_terrainData.csv"
    grid_terrain_pd = pd.read_csv(grid_terrain_file, header=0, index_col=1)

    print("modis_all_size: ", modis_all_pd.shape)
    print("station size: ", station_cell_mapper_pd.shape)
    print("training_feature_pd size: ", training_feature_pd.shape)
    print("testing_feature_pd size: ", testing_feature_pd.shape)

    all_training_pd = pd.DataFrame(
        columns=["cell_id", "year", "m", "doy", "ndsi", "grd", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs",
                 "lat", "lon", "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe"])
    all_training_pd = all_training_pd.reset_index()
    for index, row in modis_all_pd.iterrows():
        dt = datetime.strptime(row['date'], '%Y-%m-%d')
        month = dt.month
        year = dt.year
        doy = dt.timetuple().tm_yday
        print(f"Dealing {year} {doy}")
        for i in range(3, len(row.index)):
            cell_id = row.index[i][:-2]
            if cell_id in train_labels_pd.index and row['date'] in train_labels_pd:
                ndsi = row.values[i]
                swe = train_labels_pd.loc[cell_id, row['date']]
                grd = sentinel1_all_pd.loc[index, cell_id]
                eto = gridmet_eto_all_pd.loc[index, cell_id]
                pr = gridmet_pr_all_pd.loc[index, cell_id]
                rmax = gridmet_rmax_all_pd.loc[index, cell_id]
                rmin = gridmet_rmin_all_pd.loc[index, cell_id]
                tmmn = gridmet_tmmn_all_pd.loc[index, cell_id]
                tmmx = gridmet_tmmx_all_pd.loc[index, cell_id]
                vpd = gridmet_vpd_all_pd.loc[index, cell_id]
                vs = gridmet_vs_all_pd.loc[index, cell_id]
                lat = grid_terrain_pd.loc[cell_id, "Longitude [deg]"]
                lon = grid_terrain_pd.loc[cell_id, "Latitude [deg]"]
                elevation = grid_terrain_pd.loc[cell_id, "Elevation [m]"]
                aspect = grid_terrain_pd.loc[cell_id, "Aspect [deg]"]
                curvature = grid_terrain_pd.loc[cell_id, "Curvature [ratio]"]
                slope = grid_terrain_pd.loc[cell_id, "Slope [deg]"]
                eastness = grid_terrain_pd.loc[cell_id, "Eastness [unitCirc.]"]
                northness = grid_terrain_pd.loc[cell_id, "Northness [unitCirc.]"]

                if not np.isnan(swe):
                    json_kv = {"cell_id": cell_id, "year": year, "m": month, "doy": doy, "ndsi": ndsi, "grd": grd,
                               "eto": eto,
                               "pr": pr, "rmax": rmax, "rmin": rmin, "tmmn": tmmn, "tmmx": tmmx, "vpd": vpd, "vs": vs,
                               "lat": lat,
                               "lon": lon, "elevation": elevation, "aspect": aspect, "curvature": curvature,
                               "slope": slope,
                               "eastness": eastness, "northness": northness, "swe": swe}
                    # print(json_kv)
                    all_training_pd = all_training_pd.append(json_kv, ignore_index=True)

    print(all_training_pd.shape)
    all_training_pd.to_csv(all_ready_file)


def loc_closest_gridcell_id(find_lat, find_lon, valid_cols):
    grid_terrain_file = f"{github_dir}/data/terrain/gridcells_terrainData.csv"
    grid_lat_lon = pd.read_csv(grid_terrain_file, header=0, usecols=['cell_id', 'Latitude [deg]', 'Longitude [deg]']).loc[lambda df: df['cell_id'].isin(valid_cols)]
    # print(grid_lat_lon.shape)
    # print(grid_lat_lon)
    grid_lat_lon_npy = grid_lat_lon.to_numpy()
    grid_lat_lon_rad = np.array([[radians(x[2]), radians(x[1])] for x in grid_lat_lon_npy])
    ball_tree = sk.BallTree(grid_lat_lon_rad, metric="haversine")

    dist, ind = ball_tree.query([(radians(find_lat), radians(find_lon))], return_distance=True)
    # print(dist)
    print(ind[0][0])
    print("cell id: ", grid_lat_lon.iloc[ind[0][0]]['cell_id'])
    return ind[0][0], grid_lat_lon.iloc[ind[0][0]]['cell_id']


def prepare_training_csv_nsidc():
    """
  gridMET model:
    input columns: [m, doy, tmmn, tmmx, pr, vpd, eto, rmax, rmin, vs]
    output column: [swe]
  """
    all_ready_file = f"{github_dir}/data/ready_for_training/all_ready_new.csv"
    if os.path.isfile(all_ready_file):
        print("The file already exists. Exiting..")
        return
    all_gridmet_eto_file = f"{github_dir}/data/ready_for_training/gridmet_eto_all.csv"
    gridmet_eto_all_pd = pd.read_csv(all_gridmet_eto_file, header=0, index_col=0)
    all_gridmet_pr_file = f"{github_dir}/data/ready_for_training/gridmet_pr_all.csv"
    gridmet_pr_all_pd = pd.read_csv(all_gridmet_pr_file, header=0, index_col=0)
    all_gridmet_rmax_file = f"{github_dir}/data/ready_for_training/gridmet_rmax_all.csv"
    gridmet_rmax_all_pd = pd.read_csv(all_gridmet_rmax_file, header=0, index_col=0)
    all_gridmet_rmin_file = f"{github_dir}/data/ready_for_training/gridmet_rmin_all.csv"
    gridmet_rmin_all_pd = pd.read_csv(all_gridmet_rmin_file, header=0, index_col=0)
    all_gridmet_tmmn_file = f"{github_dir}/data/ready_for_training/gridmet_tmmn_all.csv"
    gridmet_tmmn_all_pd = pd.read_csv(all_gridmet_tmmn_file, header=0, index_col=0)
    all_gridmet_tmmx_file = f"{github_dir}/data/ready_for_training/gridmet_tmmx_all.csv"
    gridmet_tmmx_all_pd = pd.read_csv(all_gridmet_tmmx_file, header=0, index_col=0)
    all_gridmet_vpd_file = f"{github_dir}/data/ready_for_training/gridmet_vpd_all.csv"
    gridmet_vpd_all_pd = pd.read_csv(all_gridmet_vpd_file, header=0, index_col=0)
    all_gridmet_vs_file = f"{github_dir}/data/ready_for_training/gridmet_vs_all.csv"
    gridmet_vs_all_pd = pd.read_csv(all_gridmet_vs_file, header=0, index_col=0)
    all_nsidc_file = f"{github_dir}/data/sim_training/nsidc/2019nsidc_data.csv"
    nsidc_all_pd = pd.read_csv(all_nsidc_file, header=0, index_col=0)

    # print(nsidc_all_pd.shape)
    # print(nsidc_all_pd)

    grid_terrain_file = f"{github_dir}/data/terrain/gridcells_terrainData.csv"
    grid_terrain_pd = pd.read_csv(grid_terrain_file, header=0, index_col=0)

    # print(grid_terrain_pd.shape)
    # print(grid_terrain_pd)

    print("station size: ", station_cell_mapper_pd.shape)
    print("training_feature_pd size: ", training_feature_pd.shape)
    print("testing_feature_pd size: ", testing_feature_pd.shape)
    all_valid_columns = gridmet_eto_all_pd.columns.values
    all_training_pd = pd.DataFrame(
        columns=["cell_id", "year", "m", "day", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs", "lat", "lon",
                 "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe_0719", "depth_0719", "swe_snotel"])
    all_training_pd = all_training_pd.reset_index()
    for index, row in nsidc_all_pd.iterrows():
        month = row['Month']
        year = row['Year']
        day = row['Day']
#         print(f"Dealing {year} {month} {day}")
        lat = row['Lat']
        lon = row['Lon']
#         print("lat lon: ", lat, " ", lon)
        ind, cell_id = loc_closest_gridcell_id(lat, lon, all_valid_columns)
        swe = row['SWE']
        depth = row['Depth']
        index = index % 365
        eto = gridmet_eto_all_pd.iloc[index][cell_id]
        pr = gridmet_pr_all_pd.iloc[index][cell_id]
        rmax = gridmet_rmax_all_pd.iloc[index][cell_id]
        rmin = gridmet_rmin_all_pd.iloc[index][cell_id]
        tmmn = gridmet_tmmn_all_pd.iloc[index][cell_id]
        tmmx = gridmet_tmmx_all_pd.iloc[index][cell_id]
        vpd = gridmet_vpd_all_pd.iloc[index][cell_id]
        vs = gridmet_vs_all_pd.iloc[index][cell_id]
        lat = grid_terrain_pd.loc[ind, "Latitude [deg]"]
        lon = grid_terrain_pd.loc[ind, "Longitude [deg]"]
        elevation = grid_terrain_pd.loc[ind, "Elevation [m]"]
        aspect = grid_terrain_pd.loc[ind, "Aspect [deg]"]
        curvature = grid_terrain_pd.loc[ind, "Curvature [ratio]"]
        slope = grid_terrain_pd.loc[ind, "Slope [deg]"]
        eastness = grid_terrain_pd.loc[ind, "Eastness [unitCirc.]"]
        northness = grid_terrain_pd.loc[ind, "Northness [unitCirc.]"]
        cdate = datetime(year=int(year), month=int(month), day=int(day))
        current_date = cdate.strftime("%Y-%m-%d")
        
        if cell_id in train_labels_pd.index and current_date in train_labels_pd.columns:
#           print("Check one value: ", train_labels_pd.loc[cell_id][current_date])
          swe_snotel = train_labels_pd.loc[cell_id][current_date]
        else:
          swe_snotel = -1
#           print("Key not existed")

        if not np.isnan(swe):
            json_kv = {"cell_id":cell_id,"year":year, "m":month, "day": day, "eto":eto, "pr":pr, "rmax":rmax, "rmin":rmin, "tmmn":tmmn, "tmmx":tmmx, "vpd":vpd, "vs":vs, "lat":lat, "lon":lon, "elevation":elevation, "aspect":aspect, "curvature":curvature, "slope":slope, "eastness":eastness, "northness":northness, "swe_0719":swe, "depth_0719":depth, "swe_snotel": swe_snotel}
#             print(json_kv)
            all_training_pd = all_training_pd.append(json_kv, ignore_index=True)
#             print(all_training_pd.shape)

    print(all_training_pd.shape)
    all_training_pd.to_csv(all_ready_file)

    """
  grd_all_pd = pd.DataFrame(columns=["year", "m", "doy", "grd", "swe"])
  grd_all_pd = grd_all_pd.reset_index()
  for index, row in sentinel1_all_pd.iterrows():
    dt = datetime.strptime(row['date'], '%Y-%m-%d')
    year = dt.year
    month = dt.month
    doy = dt.timetuple().tm_yday
    for i in range(3,len(row.index)):
      cell_id = row.index[i]
      grd = row.values[i]
      if not np.isnan(grd) and cell_id in train_labels_pd.index and row['date'] in train_labels_pd:
        swe = train_labels_pd.loc[cell_id, row['date']]
        if not np.isnan(swe):
          print([month, doy, grd, swe])
          grd_all_pd = grd_all_pd.append({"year": year, "m":month, "doy": doy, "grd": grd, "swe": swe}, ignore_index = True)
  
  print(grd_all_pd.shape)
  grd_all_pd.to_csv(f"{github_dir}/data/ready_for_training/sentinel1_ready.csv")
  """


# exit() # done already

# integrate_modis()
# integrate_sentinel1()
# integrate_gridmet()
# prepare_training_csv()
prepare_training_csv_nsidc()
