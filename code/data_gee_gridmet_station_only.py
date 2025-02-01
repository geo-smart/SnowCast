import os
import glob
import re
import urllib.request
from datetime import date, datetime

import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
import dask.dataframe as dd
from dask.delayed import delayed

from snowcast_utils import homedir, snowcast_github_dir, work_dir, train_start_date, train_end_date, data_dir, gridcells_file, stations_file, all_training_points_with_station_and_non_station_file, all_training_points_with_snotel_ghcnd_file, gridcells_outfile, stations_outfile, supplement_point_for_correction_file

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
end_date = datetime.strptime(train_end_date, "%Y-%m-%d")

year_list = [start_date.year + i for i in range(end_date.year - start_date.year + 1)]

# working_dir = work_dir
print("work_dir = ", work_dir)
#stations = pd.read_csv(f'{work_dir}/station_cell_mapping.csv')
all_training_points_with_snotel_ghcnd_file = f"{work_dir}/all_training_points_snotel_ghcnd_in_westus.csv"
gridmet_save_location = f'{work_dir}/gridmet_climatology'
final_merged_csv = f"{work_dir}/salt_pepper_point_gridmet_training.csv"
variables_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'etr', 'rmax', 'rmin', 'vs']


def get_files_in_directory(gridmet_save_location):
    f = list()
    for files in glob.glob(gridmet_save_location + "/*.nc"):
        f.append(files)
    return f


def download_file(url, save_location):
    try:
        print("download_file")
        with urllib.request.urlopen(url) as response:
            file_content = response.read()
        file_name = os.path.basename(url)
        save_path = os.path.join(save_location, file_name)
        with open(save_path, 'wb') as file:
            file.write(file_content)
        print(f"File downloaded successfully and saved as: {save_path}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {str(e)}")


def download_gridmet_climatology():
    folder_name = gridmet_save_location
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    base_metadata_url = "http://www.northwestknowledge.net/metdata/data/"
    

    for var in variables_list:
        for y in year_list:
            download_link = base_metadata_url + var + '_' + '%s' % y + '.nc'
            print("downloading", download_link)
            if not os.path.exists(os.path.join(folder_name, var + '_' + '%s' % y + '.nc')):
                download_file(download_link, folder_name)


def process_station(ds, lat, lon):
    subset_data = ds.sel(lat=lat, lon=lon, method='nearest')
    subset_data['lat'] = lat
    subset_data['lon'] = lon
    converted_df = subset_data.to_dataframe()
    converted_df = converted_df.reset_index(drop=False)
    converted_df = converted_df.drop('crs', axis=1)
    return converted_df

def get_year_short_var_name_from_path(file_path):
    # Regular expression to match the variable name and the year
    pattern = '/([^/]*)_([0-9]{4})\\.nc$'
    match = re.search(pattern, file_path)
    if match:
        variable_name = match.group(1)
        year = match.group(2)
        return year, variable_name
    else:
        print("No match found.")
        return None, None

def get_gridmet_variable(gridmet_nc_file_name, target_points_csv_path):
    print(f"Reading values from {gridmet_nc_file_name}")
    ds = xr.open_dataset(gridmet_nc_file_name)
    var_name = list(ds.keys())[0]
    year, short_var_name = get_year_short_var_name_from_path(gridmet_nc_file_name)

    # Step 2: Define the start and end of that year
    year_start = datetime(int(year), 1, 1)
    year_end = datetime(int(year), 12, 31)

    # Step 3: Check if there is an overlap between the year and the date range
    if max(start_date, year_start) <= min(end_date, year_end):
        print(f"The year {year} is within the range.")
    else:
        print(f"The year {year} is not within the range. exiting")
        return

    file_name = os.path.basename(target_points_csv_path)
    csv_file = f'{gridmet_save_location}/{file_name}_{year}_{short_var_name}_gridmet_training.csv'
    if os.path.exists(csv_file):
        print(f"The file '{csv_file}' exists.")
        return

    result_data = []
    # stations = pd.read_csv(all_training_points_with_snotel_ghcnd_file)
    stations = pd.read_csv(target_points_csv_path)
    for _, row in stations.iterrows():
        delayed_process_data = delayed(process_station)(ds, row['latitude'], row['longitude'])
        result_data.append(delayed_process_data)

    print("ddf = dd.from_delayed(result_data)")
    ddf = dd.from_delayed(result_data)
    
    print("result_df = ddf.compute()")
    result_df = ddf.compute()
    result_df.to_csv(csv_file, index=False)
    print(f'Completed extracting data for {gridmet_nc_file_name}')


def parse_var_year_from_path(file_path):
    # Regular expression to extract the year and variable name
    regex_pattern = "_(\d{4})_(.*?)_gridmet_training"
    match = re.search(regex_pattern, file_path)
    if match:
        year = match.group(1)
        variable_name = match.group(2)
        return year, variable_name
    else:
        # print("Don't match")
        return None, None

def merge_similar_variables_from_different_years(target_points_file_path):
    print(f"Listing files in {gridmet_save_location}")
    files = os.listdir(gridmet_save_location)
    file_groups = {}

    point_file_name = os.path.basename(target_points_file_path)

    for filename in files:
        year, variable_name = parse_var_year_from_path(filename)
        if year is None:
            continue;
        
        print(filename)
        file_groups.setdefault(variable_name, []).append(filename)
        # base_name, year_ext = os.path.splitext(filename)
        # parts = base_name.split('_')
        # print(parts)
        # print(year_list)
        # if len(parts) == 4 and parts[3] == "ghcnd" and year_ext == '.csv' and parts[1].isdigit() and int(parts[1]) in year_list:
        #     file_groups.setdefault(parts[0], []).append(filename)

    print("file_groups = ", file_groups)
    for variable_name, file_list in file_groups.items():
        if len(file_list) > 1:
            dfs = []
            for filename in file_list:
                df = pd.read_csv(os.path.join(gridmet_save_location, filename))
                dfs.append(df)
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_filename = f"{point_file_name}_{variable_name}_merged.csv"
            merged_df.to_csv(
                os.path.join(gridmet_save_location, merged_filename), 
                index=False
            )
            print(f"Merged {file_list} into {merged_filename}")


def merge_all_variables_together(target_points_csv_path):
    merged_df = None
    file_paths = []

    point_file_name = os.path.basename(target_points_csv_path)

    for filename in os.listdir(gridmet_save_location):
        if filename.endswith("_merged.csv") and point_file_name in filename:
            file_paths.append(os.path.join(gridmet_save_location, filename))

    print("file_paths = ", file_paths)
	
    rmin_merged_path = os.path.join(gridmet_save_location, f'{point_file_name}_rmin_merged.csv')
    rmax_merged_path = os.path.join(gridmet_save_location, f'{point_file_name}_rmax_merged.csv')
    tmmn_merged_path = os.path.join(gridmet_save_location, f'{point_file_name}_tmmn_merged.csv')
    tmmx_merged_path = os.path.join(gridmet_save_location, f'{point_file_name}_tmmx_merged.csv')

    print("saving to ", rmin_merged_path)
    
    df_rmin = pd.read_csv(rmin_merged_path)
    df_rmax = pd.read_csv(rmax_merged_path)
    df_tmmn = pd.read_csv(tmmn_merged_path)
    df_tmmx = pd.read_csv(tmmx_merged_path)
    
    df_rmin.rename(columns={'relative_humidity': 'relative_humidity_rmin'}, inplace=True)
    df_rmax.rename(columns={'relative_humidity': 'relative_humidity_rmax'}, inplace=True)
    df_tmmn.rename(columns={'air_temperature': 'air_temperature_tmmn'}, inplace=True)
    df_tmmx.rename(columns={'air_temperature': 'air_temperature_tmmx'}, inplace=True)
    
    df_rmin.to_csv(rmin_merged_path)
    df_rmax.to_csv(rmax_merged_path)
    df_tmmn.to_csv(tmmn_merged_path)
    df_tmmx.to_csv(tmmx_merged_path)
    
    if file_paths:
        merged_df = pd.read_csv(file_paths[0])
        for file_path in file_paths[1:]:
            df = pd.read_csv(file_path)
            merged_df = pd.concat([merged_df, df], axis=1)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        final_merged_csv = f"{target_points_csv_path}_gridmet_training.csv"
        merged_df.to_csv(final_merged_csv, index=False)
        print(f"all files are saved to {final_merged_csv}")


if __name__ == "__main__":
    
    download_gridmet_climatology()
    
    # mock out as this takes too long
    nc_files = get_files_in_directory(gridmet_save_location)
    for nc in nc_files:
        # should check if the nc file year number is in the year_list
        get_gridmet_variable(
            gridmet_nc_file_name = nc, 
            target_points_csv_path = supplement_point_for_correction_file
        )
    
    merge_similar_variables_from_different_years(supplement_point_for_correction_file)
    merge_all_variables_together(supplement_point_for_correction_file)

