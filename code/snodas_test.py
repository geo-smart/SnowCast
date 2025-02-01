import os
import requests
import tarfile
import gzip
import shutil
from datetime import datetime, timedelta
from osgeo import gdal
from snowcast_utils import test_start_date, test_end_date, data_dir, plot_dir, process_dates_in_range

import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

sample_nc_file = f"{data_dir}/snodas/snodas_sample.nc"
western_us_coords = f'{data_dir}/srtm/dem_file.tif.csv'
mapper_file = os.path.join(data_dir, f'snodas/snodas_to_dem_mapper.csv')

def prepare_snodas_grid_mapper(
    snodas_netcdf=sample_nc_file, 
    target_grid_csv = western_us_coords,
):
    """
    Prepares a mapper file to map coordinates between SNODAS grid coordinates and target grid coordinates.

    This function performs the following steps:
    1. Checks if the mapper file already exists. If yes, the function skips the generation process.
    2. Reads target grid coordinates from a CSV file (`target_grid_csv`) containing 'Longitude' and 'Latitude'.
    3. Uses a sample SNODAS NetCDF file (`snodas_netcdf`) to map SNODAS grid coordinates ('snodas_x' and 'snodas_y') to target grid coordinates.
    4. Saves the resulting mapper file as a CSV (`mapper_file`) containing columns 'Latitude', 'Longitude', 'snodas_x', and 'snodas_y'.

    Args:
    - snodas_netcdf (str): Path to the SNODAS NetCDF file.
    - target_grid_csv (str): Path to the CSV file containing target grid coordinates with 'Longitude' and 'Latitude'.
    - mapper_file (str): Path to save the generated mapper CSV file.

    Returns:
    - None: The function primarily generates the mapper file with side effects.
    """
    
    if os.path.exists(mapper_file):
        # print(f"The file {mapper_file} exists. Skipping generation.")
        pass
    else:
        print(f"Starting to generate {mapper_file}")
        
        # Read target grid coordinates from the CSV file
        target_grid_df = pd.read_csv(target_grid_csv, usecols=['Longitude', 'Latitude'])

        # Open SNODAS NetCDF file
        snodas_data = xr.open_dataset(snodas_netcdf)

        # Get the SNODAS grid coordinates (longitude and latitude)
        snodas_lon = snodas_data['lon'].values
        snodas_lat = snodas_data['lat'].values

        # Create a grid of SNODAS coordinates
        snodas_coords = np.array(np.meshgrid(snodas_lon, snodas_lat)).T.reshape(-1, 2)

        # Build a KDTree for SNODAS coordinates
        snodas_kdtree = cKDTree(snodas_coords)

        # Get target coordinates from the dataframe
        target_coords = target_grid_df[['Longitude', 'Latitude']].values

        # Apply nearest neighbor search using KDTree
        _, indices = snodas_kdtree.query(target_coords)

        # Map target grid coordinates to SNODAS grid coordinates
        target_grid_df['snodas_lon'] = snodas_coords[indices, 0]
        target_grid_df['snodas_lat'] = snodas_coords[indices, 1]

        # Save the mapper file as CSV
        print(f"Saving mapper CSV file: {mapper_file}")
        target_grid_df.to_csv(mapper_file, index=False, columns=['Latitude', 'Longitude', 'snodas_lon', 'snodas_lat'])

def create_envi_header(dat_file_path):
    """
    Creates an ENVI header file for the corresponding .dat file.
    """
    hdr_file_path = dat_file_path.replace(".dat", ".hdr")
    if not os.path.exists(hdr_file_path):
        header_content = """ENVI
samples = 6935
lines = 3351
bands = 1
header offset = 0
file type = ENVI Standard
data type = 2
interleave = bsq
byte order = 1
"""
        with open(hdr_file_path, "w") as hdr_file:
            hdr_file.write(header_content)
        print(f"Created header file: {hdr_file_path}")
    else:
        print(f"Header file {hdr_file_path} already exists.")

def download_and_extract_tar(url, download_dir, extract_dir):
    tar_file_name = url.split("/")[-1]
    tar_file_path = os.path.join(download_dir, tar_file_name)

    if not os.path.exists(tar_file_path):
        print(f"Downloading {url} to {tar_file_path}")
        response = requests.get(url)
        response.raise_for_status()
        with open(tar_file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {tar_file_name}")
    # else:
    #     print(f"Tar file {tar_file_name} already exists, skipping download.")
    
    extracted_folder = os.path.join(extract_dir, tar_file_name.replace(".tar", ""))
    if not os.path.exists(extracted_folder):
        with tarfile.open(tar_file_path, "r") as tar_ref:
            tar_ref.extractall(extracted_folder)
        print(f"Extracted {tar_file_name} to {extracted_folder}")
    # else:
    #     print(f"Folder {extracted_folder} already exists, skipping extraction.")
    return extracted_folder

def decompress_gz(file_path, output_dir):
    file_name = os.path.basename(file_path).replace(".gz", "")
    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        with gzip.open(file_path, "rb") as gz_file:
            with open(output_path, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)
        print(f"Decompressed {file_path} to {output_path}")
    # else:
    #     print(f"File {output_path} already exists, skipping decompression.")
    return output_path

def convert_to_netcdf(dat_file_path, output_dir):
    netcdf_file_path = os.path.join(output_dir, os.path.basename(dat_file_path).replace(".dat", ".nc"))
    if os.path.exists(netcdf_file_path):
        # print(f"{netcdf_file_path} exists. skip.")
        pass
    else:
        dataset = gdal.Open(dat_file_path)
        if dataset is None:
            print(f"Failed to open {dat_file_path}. File format not supported by GDAL.")
            return
        # Define spatial extent and projection
        spatial_extent = "-124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000"
        projection = "EPSG:4326"
        gdal.Translate(
            netcdf_file_path, 
            dataset, 
            format="NetCDF",
            outputSRS=projection,
            noData=-9999,
            outputBounds=(
                float(spatial_extent.split()[0]), 
                float(spatial_extent.split()[1]),
                float(spatial_extent.split()[2]), 
                float(spatial_extent.split()[3])
            )
        )
        print(f"Converted {dat_file_path} to {netcdf_file_path}")
    return netcdf_file_path

def process_snodas_file(netcdf_file, mapper_file, current_date, output_dir):
    """
    Process a SNODAS NetCDF file, extract values for specific coordinates based on the provided mapper CSV file,
    and save the result in a new CSV file.

    Parameters:
    - netcdf_file (str): Path to the SNODAS NetCDF file to be processed.
    - mapper_file (str): Path to the mapper CSV file that contains target grid coordinates and SNODAS grid coordinates.
    - current_date (str): Current date to be associated with the processed data.
    - output_dir (str): Directory to save the resulting CSV file.

    Returns:
    - str: Path to the saved CSV file containing the processed data.
    """
    # Output file path
    output_file = os.path.join(output_dir, f"{current_date}_snodas_output.csv")
    print(f"Saving processed data to: {output_file}")
    if os.path.exists(output_file):
        # print(f"{output_file} exists. skip.")
        pass
    else:
        # Read the target grid coordinates and SNODAS grid mapping from the mapper CSV file
        station_df = pd.read_csv(mapper_file)
        print(f"Opening SNODAS NetCDF file: {netcdf_file}")
        
        # Open the SNODAS NetCDF file using xarray
        snodas_data = xr.open_dataset(netcdf_file)
        
        # Extract SNODAS grid coordinates (lon and lat) and the snow cover data (Band1)
        snodas_lon = snodas_data['lon'].values
        snodas_lat = snodas_data['lat'].values
        snodas_values = snodas_data['Band1'].values  # assuming Band1 contains snow cover data
        
        # Function to map SNODAS grid coordinates to the extracted data
        def get_snodas_value(row):
            # Target grid coordinates from mapper
            target_lon = row['Longitude']
            target_lat = row['Latitude']
            
            # Find the closest matching SNODAS grid coordinate for the target grid
            lon_idx = (np.abs(snodas_lon - target_lon)).argmin()
            lat_idx = (np.abs(snodas_lat - target_lat)).argmin()
            
            # Extract the snodas value at the closest coordinates
            snodas_value = snodas_values[lat_idx, lon_idx]
            
            # Handle cases where the value is the NoData value (-9999) and convert it to NaN
            if snodas_value == -9999:
                return np.nan
            return snodas_value

        # Apply the function to get snow cover data for each row in the target grid
        station_df['snodas'] = station_df.apply(get_snodas_value, axis=1)
        
        # Filter out rows with missing or NaN snodas values
        valid_data = station_df[station_df['snodas'].notna()]
        
        # Add the current date to the final dataset
        valid_data['date'] = current_date
        
        # Save the result to a CSV
        valid_data.to_csv(output_file, index=False, columns=['date', 'Latitude', 'Longitude', 'snodas'])
    
    return output_file

def snodas_callback(current_date):
    current_date_str = current_date.strftime("%Y-%m-%d")

    # Define directories
    download_dir = os.path.join(data_dir, "snodas/raw/")
    extract_dir = os.path.join(data_dir, "snodas/extracted/")
    output_dir = os.path.join(data_dir, "snodas/netcdf/")
    csv_dir = os.path.join(data_dir, "snodas/csv/")

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    final_output_csv_file = os.path.join(output_dir, f"{current_date_str}_snodas_output.csv")
    if os.path.exists(final_output_csv_file):
        return final_output_csv_file

    prepare_snodas_grid_mapper()

    # URL for tar files
    # Extract the year and month in the required format
    day = current_date.day
    month_num = current_date.month  # Numeric month (1-12)
    month = current_date.strftime("%b")  # Month abbreviated (e.g., Jan, Feb, Mar)
    
    # Construct the updated URL with day and month swapped
    base_url = f"https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{current_date.year}/{month_num:02d}_{month.capitalize()}/"
    
    
    # Dates for the past 7 days
    dates = [(current_date - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
    tar_file = f"SNODAS_{current_date.strftime('%Y%m%d')}.tar"

    tar_url = os.path.join(base_url, tar_file)
    extracted_folder = download_and_extract_tar(tar_url, download_dir, extract_dir)

    # Process .gz files inside the extracted folder
    for root, _, files in os.walk(extracted_folder):
        for file in files:
            if file.endswith(".dat.gz") and "1034" in file:
                gz_file_path = os.path.join(root, file)
                decompressed_file = decompress_gz(gz_file_path, root)
                if decompressed_file.endswith(".dat") and "1034" in decompressed_file:
                    create_envi_header(decompressed_file)
                    netcdf_file = convert_to_netcdf(
                        decompressed_file, output_dir
                    )
                    process_snodas_file(
                        netcdf_file, mapper_file, current_date_str, csv_dir
                    )
    return final_output_csv_file

if __name__ == "__main__":
    process_dates_in_range(
        # start_date=test_start_date,
        # end_date=test_end_date,
        start_date="2000-01-01",
        end_date="2015-12-31",
        callback=snodas_callback,
    )

