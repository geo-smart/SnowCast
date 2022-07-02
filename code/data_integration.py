# Integrate all the datasets into one training dataset
import os
import os.path
from datetime import datetime

import numpy as np
import pandas as pd

print("Integrating all datasets into one dataset:")  # logging

# Directories
homedir                         = os.path.expanduser('~')
github_dir                      = f"{homedir}/Documents/GitHub/SnowCast"
gridcells_file                  = f"{github_dir}/data/snowcast_provided/grid_cells.geojson"
model_dir                       = f"{github_dir}/model/"

training_feature_file           = f"{github_dir}/data/snowcast_provided/ground_measures_train_features.csv"
testing_feature_file            = f"{github_dir}/data/snowcast_provided/ground_measures_test_features.csv"
train_labels_file               = f"{github_dir}/data/snowcast_provided/train_labels.csv"
ground_measure_metadata_file    = f"{github_dir}/data/snowcast_provided/ground_measures_metadata.csv"

station_cell_mapper_file        = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"

all_mod_file                    = f"{github_dir}/data/ready_for_training/modis_all.csv"
all_sentinel1_file              = f"{github_dir}/data/ready_for_training/sentinel1_all.csv"
all_sentinel2_file              = f"{github_dir}/data/ready_for_training/" + "sentinel2_{var}_all.csv"  # to be formatted
all_gridmet_file                = f"{github_dir}/data/ready_for_training/" + "gridmet_{var}_all.csv"  # to be formatted


print(f"\t{homedir=}")  # logging
print(f"\t{github_dir=}")  # logging


# Pandas dataframes
training_feature_pd = pd.read_csv(training_feature_file, header=0, index_col=0)
testing_feature_pd = pd.read_csv(testing_feature_file, header=0, index_col=0)
train_labels_pd = pd.read_csv(train_labels_file, header=0, index_col=0)

station_cell_mapper_pd = pd.read_csv(station_cell_mapper_file, header=0, index_col=0)


def get_date_str(x):
    return x.split(" ")[0]


def integrate_modis():
    """
    Integrate all MODIS data into mod_all.csv
    """
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

    dates = pd.date_range(start='1/1/2013', end='12/31/2021', freq='D').astype(str)

    # print(mod_all_df.head())
    var_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']

    for var in var_list:
        gridmet_all_df = pd.DataFrame(columns=["date"])
        gridmet_all_df['date'] = dates
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

def integrate_sentinel2():
    """
    Integrate all sentinel2 data into sentinel.csv
    """

    dates = pd.date_range(start='2017-03-28', end='12/31/2021', freq='D').astype(str)

    # print(mod_all_df.head())
    var_list = ['swir_1', 'swir_2', 'blue']

    for var in var_list:
        sentinel2_all_df = pd.DataFrame(columns=["date"])
        sentinel2_all_df['date'] = dates
        if os.path.isfile(all_sentinel2_file):
            return
        for ind in station_cell_mapper_pd.index:
            current_cell_id = station_cell_mapper_pd["cell_id"][ind]
            print(current_cell_id)
            sentinel2_single_file = f"{github_dir}/data/sim_training/sentinel2/{var}_{current_cell_id}.csv"
            if os.path.isfile(sentinel2_single_file) and current_cell_id not in sentinel2_all_df:
                sentinel2_single_pd = pd.read_csv(sentinel2_single_file, header=0)
                sentinel2_single_pd = sentinel2_single_pd[["date", var]]
                sentinel2_single_pd = sentinel2_single_pd.rename(columns={var: current_cell_id})
                # sentinel1_single_pd['date'] = sentinel1_single_pd['date'].astype('datetime64[ns]')
                sentinel2_single_pd['date'] = pd.to_datetime(sentinel2_single_pd['date']).dt.round("D").astype(str)
                print("sentinel2_single_pd: ", sentinel2_single_pd.head())
                print("sentinel2_single_pd check value: ", sentinel2_single_pd[sentinel2_single_pd["date"] == "2015-04-01"])
                sentinel2_single_pd = sentinel2_single_pd.drop_duplicates(subset=['date'],
                                                                      keep='first')  # this will remove all the other values of the same day

                sentinel2_all_df = pd.merge(sentinel2_all_df, sentinel2_single_pd, how='left', on="date")
                print("sentinel2_all_df check value: ", sentinel2_all_df[sentinel2_all_df["date"] == "2015-04-01"])
                print("sentinel2_all_df: ", sentinel2_all_df.shape)

        print(sentinel2_all_df.shape)
        sentinel2_all_df.to_csv(all_sentinel2_file)


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
    Sentinel2 model:
      input columns: [m, doy, swir_1, swir_2, blue]
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

    all_sentinel2_swir_1_file = f"{github_dir}/data/ready_for_training/sentinel2_swir_1_all.csv"
    sentinel2_swir_1_all_pd = pd.read_csv(all_sentinel2_swir_1_file, header=0, index_col=0)
    all_sentinel2_swir_2_file = f"{github_dir}/data/ready_for_training/sentinel2_swir_2_all.csv"
    sentinel2_swir_2_all_pd = pd.read_csv(all_sentinel2_swir_2_file, header=0, index_col=0)
    all_sentinel2_blue_file = f"{github_dir}/data/ready_for_training/sentinel2_blue_all.csv"
    sentinel2_blue_all_pd = pd.read_csv(all_sentinel2_blue_file, header=0, index_col=0)

    grid_terrain_file = f"{github_dir}/data/terrain/gridcells_terrainData.csv"
    grid_terrain_pd = pd.read_csv(grid_terrain_file, header=0, index_col=1)

    print("modis_all_size: ", modis_all_pd.shape)
    print("station size: ", station_cell_mapper_pd.shape)
    print("training_feature_pd size: ", training_feature_pd.shape)
    print("testing_feature_pd size: ", testing_feature_pd.shape)

    all_training_pd = pd.DataFrame(
        columns=["cell_id", "year", "m", "doy", "ndsi", "grd", "eto", "pr", "rmax", "rmin", "tmmn", "tmmx", "vpd", "vs",
                 "lat", "lon", "elevation", "aspect", "curvature", "slope", "eastness", "northness", "swe", "swir_1", "swir_2", "blue"])
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
                swir_1 = sentinel2_swir_1_all_pd.loc[index, cell_id]
                swir_2 = sentinel2_swir_2_all_pd.loc[index, cell_id]
                blue = sentinel2_blue_all_pd.loc[index, cell_id]
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
                               "eto": eto, "pr": pr, "rmax": rmax, "rmin": rmin, "tmmn": tmmn, "tmmx": tmmx, "vpd": vpd,
                               "vs": vs, "swir_1": swir_1, "swir_2": swir_2, "blue": blue, "lat": lat, "lon": lon, "elevation": elevation, "aspect": aspect,
                               "curvature": curvature, "slope": slope, "eastness": eastness, "northness": northness,
                               "swe": swe, }
                    # print(json_kv)
                    all_training_pd = all_training_pd.append(json_kv, ignore_index=True)

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

integrate_modis()
# integrate_sentinel1()
# integrate_gridmet()
# prepare_training_csv()
