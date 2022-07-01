import tempfile

import geojson
import geopandas as gpd
import pandas as pd
import requests
import xarray as xr


# Not used directly, but used via xarray


def get_range(index, variable):
    # You can see it has a 1-indexed base line number, staring byte position, date, variable, atmosphere level,
    # and forecast description. The lines are colon-delimited. 

    # Let's grab surface temperature `TMP:surface`.
    sfc_var_idx = [l for l in index if variable in l][0].split(":")
    print("Surface temp line:", sfc_var_idx)

    # Pluck the byte offset from this line, plus the beginning offset of the next line
    line_num = int(sfc_var_idx[0])
    range_start = sfc_var_idx[1]

    # The line number values are 1-indexed, so we don't need to increment it to get the next list index,
    # but check we're not already reading the last line
    next_line = index[line_num].split(':') if line_num < len(index) else None

    # Pluck the start of the next byte offset, or nothing if we were on the last line
    range_end = next_line[1] if next_line else None

    print(f"Byte range: {range_start}-{range_end}")
    return range_start, range_end


# Constants for creating the full URL
blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
sector = "conus"

# cycle = 12         # noon, 
cycle = 00  # CC is the model cycle runtime (i.e. 00, 01, 02, 03)
# forecast_hour = 1   # offset from cycle time
product = "wrfsfcf"  # 2D surface levels

# Put it all together
# file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
file_path = f"hrrr.t{cycle:02}z.{product}00.grib2"
# yesterday = date.today() - timedelta(days=1)

# Take a peek at the content of the index
# print(*idx[0:30], sep="\n")

data_dir = '../../SnowCast/data/'
out_dir = data_dir + 'HRRR/'
# gridcells_file = data_dir + 'station_gridcell/grid_cells.geojson'
gridcells_file = '/home/jovyan/SWE/SnowCast/data/station_gridcell/grid_cells.geojson'
station_file = data_dir + 'station_gridcell/ground_measures_metadata.csv'

# Load metadata
gridcellsGPD = gpd.read_file(gridcells_file)
gridcells = geojson.load(open(gridcells_file))
station = pd.read_csv(station_file)

use_proj = gridcellsGPD.crs

variable_list = [":TMP:surface", ":ASNOW:surface", ":SNOWC:surface", ":SNOD:surface", ":WEASD:surface"]
variable_names = ["t", 'asnow', 'snowc', 'sde', 'sdwe']
K_to_C = -273.15

# version 1 - 30 Sept 2014
# version 2 - 23 Aug 2016
# version 3 - 12 July 2018
# version 4 - IMPLEMENTED 12z Wed 2 Dec 2020 

start_date = "2021-10-01"
end_date = "2021-12-31"

# instantiate output panda dataframes
df_gridcells = pd.DataFrame(columns=("cell_id", "date", "value"))
df_station = pd.DataFrame(columns=("station_id", "date", "value"))

# for each grid cell
gridcells_outfile = out_dir + 'gridcells_meteo.csv'
station_outfile = out_dir + "station_meteo.csv"

for idx_var in range(0, len(variable_list)):

    gridcells_outfile = out_dir + 'gridcells_meteo' + variable_list[idx_var] + '.csv'
    df_all = []

    for idx_date in pd.date_range(start=start_date, end=end_date):
        url = f"{blob_container}/hrrr.{idx_date:%Y%m%d}/{sector}/{file_path}"
        print(url)
        # Fetch the idx file by appending the .idx file extension to our already formatted URL
        r = requests.get(f"{url}.idx")
        idx_data = r.text.splitlines()

        for idx_cell in range(0, len(gridcellsGPD)):

            gridcell_boundary = gridcellsGPD.geometry[idx_cell:(idx_cell + 1)]
            lat = gridcell_boundary.centroid.y.values
            lon = gridcell_boundary.centroid.x.values

            variable = variable_list[idx_var]
            file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)

            range_start, range_end = get_range(idx_data, variable)

            headers = {"Range": f"bytes={range_start}-{range_end}"}
            resp = requests.get(url, headers=headers, stream=True)

            with file as f:
                f.write(resp.content)

            ds = xr.open_dataset(file.name, engine='cfgrib',
                                 backend_kwargs={'indexpath': ''})

            ds1 = ds.where(abs(ds.latitude - lat) == abs(ds.latitude.values - lat).min(), drop=True)
            ds2 = ds1.where(abs(ds1.latitude - lon) == abs(ds1.latitude.values - lon).min(), drop=True)
            if idx_var == 0:
                value = ds2.t.values[0][0] + K_to_C
            else:
                value = ds2[variable_names[idx_var]].values[0][0]

            temp = pd.DataFrame(
                data={"cell_id": [gridcellsGPD["cell_id"][idx_cell]], 'date': [idx_date], "value": [value]})
            if len(df_all) == 0:
                df_all = temp
            else:
                df_all = pd.concat([df_all, temp])

        df_all.to_csv(gridcells_outfile)

# for each station location

for idx_var in range(0, len(variable_list)):

    station_outfile = out_dir + 'station_meteo' + variable_list[idx_var] + '.csv'
    df_all = []

    for idx_date in pd.date_range(start=start_date, end=end_date):
        url = f"{blob_container}/hrrr.{idx_date:%Y%m%d}/{sector}/{file_path}"
        print(url)
        # Fetch the idx file by appending the .idx file extension to our already formatted URL
        r = requests.get(f"{url}.idx")
        idx_data = r.text.splitlines()

        for idx_cell in range(0, station.shape[0]):

            lat = station.latitude[idx_cell]
            lon = station.longitude[idx_cell]

            variable = variable_list[idx_var]
            file = tempfile.NamedTemporaryFile(prefix="tmp_", delete=False)

            range_start, range_end = get_range(idx_data, variable)

            headers = {"Range": f"bytes={range_start}-{range_end}"}
            resp = requests.get(url, headers=headers, stream=True)

            with file as f:
                f.write(resp.content)

            ds = xr.open_dataset(file.name, engine='cfgrib',
                                 backend_kwargs={'indexpath': ''})

            ds1 = ds.where(abs(ds.latitude - lat) == abs(ds.latitude.values - lat).min(), drop=True)
            ds2 = ds1.where(abs(ds1.latitude - lon) == abs(ds1.latitude.values - lon).min(), drop=True)
            if idx_var == 0:
                value = ds2.t.values[0][0] + K_to_C
            else:
                value = ds2[variable_names[idx_var]].values[0][0]

            temp = pd.DataFrame(
                data={"station_id": [station["station_id"][idx_cell]], 'date': [idx_date], "value": [value]})
            if len(df_all) == 0:
                df_all = temp
            else:
                df_all = pd.concat([df_all, temp])

        df_all.to_csv(station_outfile)
