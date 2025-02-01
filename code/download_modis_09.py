# Write your first Python code in Geoweaver
# Code to download 7 bands of MODIS (MOD09GA) to compute fSCA
# Code first drafted by Millie Spencer and Ziheng Sun in UW Geohackweek 2024


import os
import sys
import subprocess
import threading
from datetime import datetime, timedelta

import requests
import earthaccess
from osgeo import gdal
from snowcast_utils import date_to_julian, work_dir, homedir
import logging

log = logging.getLogger(__name__)

# change directory before running the code
shared_data_folder_path = f"{homedir}/mod09"
os.makedirs(shared_data_folder_path, exist_ok=True)
os.chdir(shared_data_folder_path)


### Define the timeframe of interest: ###
# to start with, we'll look at 1 month of data 

tile_list = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04", "h10v05", "h11v04", "h11v05", "h12v04", "h12v05",
             "h13v04", "h13v05", "h15v04", "h16v03", "h16v04", ]

### Set up storage paths ###

# The folder path where the HDF files will be temporarily stored.
input_folder = shared_data_folder_path + "/temp/"

# The folder path where the GeoTIFF files will be stored after conversion from HDF.
output_folder = shared_data_folder_path + "/output_folder/"

# The folder path where the final merged output GeoTIFF files will be stored.
modis_day_wise = shared_data_folder_path + "/final_output/"

# Create necessary directories if they do not exist.
os.makedirs(output_folder, exist_ok=True)
os.makedirs(modis_day_wise, exist_ok=True)

# List of band names you want to download
bands = [
    "MODIS_Grid_500m_2D:sur_refl_b01_1",  # Band 1
    "MODIS_Grid_500m_2D:sur_refl_b02_1",  # Band 2
    "MODIS_Grid_500m_2D:sur_refl_b03_1",  # Band 3
    "MODIS_Grid_500m_2D:sur_refl_b04_1",  # Band 4
    "MODIS_Grid_500m_2D:sur_refl_b05_1",  # Band 5
    "MODIS_Grid_500m_2D:sur_refl_b06_1",  # Band 6
    "MODIS_Grid_500m_2D:sur_refl_b07_1",  # Band 7
    "MODIS_Grid_500m_2D:QC_500m_1"        # Quality Control
]

def get_band_path_name_from_band_original_name(original_band_name):
    return original_band_name.split(':')[1]

def convert_hdf_to_geotiff(hdf_file, output_folder, force=False):
    """
    Converts a specified HDF file to a GeoTIFF format.

    Args:
        hdf_file (str): The file path of the HDF file to be converted.
        output_folder (str): The directory where the converted GeoTIFF file will be saved.

    Returns:
        None
    """
    hdf_ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    # Iterate through each band and download it
    for target_subdataset_name in bands:
        # Your existing code to download the data for each band
        # print(f"Exporting {target_subdataset_name} into geotiff")
        # (insert your downloading code here)
        
        band = get_band_path_name_from_band_original_name(target_subdataset_name)
        # Create a name for the output file based on the HDF file name and subdataset
        output_file_name = os.path.splitext(os.path.basename(hdf_file))[0] + f"_{band}.tif"
        output_path = os.path.join(output_folder, output_file_name)
    
        if os.path.exists(output_path) and not force:
            pass
        else:
            try:
                for subdataset in hdf_ds.GetSubDatasets():
                    if target_subdataset_name in subdataset[0]:
                        ds = gdal.Open(subdataset[0], gdal.GA_ReadOnly)
                        gdal.Translate(output_path, ds)
                        ds = None
                        break
            except Exception as e:
                log.exception("Something wrong with the downloaded HDF. Redownloading and try again..")
                
    
    hdf_ds = None

def convert_all_hdf_in_folder(folder_path, output_folder, force=False):
    """
    Converts all HDF files in a given folder to GeoTIFF format.

    Args:
        folder_path (str): The directory containing HDF files to be converted.
        output_folder (str): The directory where the converted GeoTIFF files will be saved.

    Returns:
        list: A list of file names that were found in the folder.
    """
    file_lst = list()
    for file in os.listdir(folder_path):
        file_lst.append(file)
        if file.lower().endswith(".hdf"):
            hdf_file = os.path.join(folder_path, file)
            convert_hdf_to_geotiff(hdf_file, output_folder, force=False)
    return file_lst

def merge_tifs(folder_path, target_date, output_file, band_name):
    """
    Merges multiple GeoTIFF files into a single GeoTIFF file for a specific date.

    Args:
        folder_path (str): The directory containing GeoTIFF files to be merged.
        target_date (datetime): The date for which the GeoTIFF files should be merged.
        output_file (str): The file path where the merged GeoTIFF file will be saved.

    Returns:
        None
    """
    julian_date = date_to_julian(target_date)
    tif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(f'{band_name}.tif') and julian_date in f]
    print("tif_files = ", tif_files)
    
    if len(tif_files) == 0:
        gdal_command = ['gdal_translate', '-b', '1', '-outsize', '100%', '100%', '-scale', '-100', '16000', '0', '255', 
                        f"{homedir}/fsca/final_output/fsca_template.tif", output_file]
        print(gdal_command)
        subprocess.run(gdal_command)
    else:
        gdal_command = ['gdalwarp', '-r', 'min'] + tif_files + [f"{output_file}_500m.tif"]
        print(gdal_command)
        subprocess.run(gdal_command)
        
        gdal_command = ['gdalwarp', '-t_srs', 'EPSG:4326', '-tr', '0.036', '0.036', '-cutline', 
                        f'{work_dir}/template.shp', 
                        '-crop_to_cutline', '-overwrite', f"{output_file}_500m.tif", output_file]
        print(gdal_command)
        subprocess.run(gdal_command)

def list_files(directory):
    """
    Lists all files in a specified directory. doc-string

    Args:
        directory (str): The directory from which to list files.

    Returns:
        list: A list of absolute file paths in the specified directory.
    """
    return [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def merge_tiles(date, hdf_files):
    """
    Merges multiple tiles into a single GeoTIFF file for a specific date.

    Args:
        date (str): The date for which the tiles should be merged (format: YYYY-MM-DD).
        hdf_files (list): A list of HDF file paths to be merged.

    Returns:
        None
    """
    path = f"data/{date}/"
    files = list_files(path)
    merged_filename = f"data/{date}/merged.tif"
    merge_command = ["gdal_merge.py", "-o", merged_filename, "-of", "GTiff"] + files
    try:
        subprocess.run(merge_command)
        print(f"Merged tiles into {merged_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging tiles: {e}")

def download_url(date, url):
    """
    Downloads a file from a specified URL to a local directory for a specific date.

    Args:
        date (str): The date for which the file is being downloaded (format: YYYY-MM-DD).
        url (str): The URL from which to download the file.

    Returns:
        None
    """
    file_name = url.split('/')[-1]
    if os.path.exists(f'data/{date}/{file_name}'):
        print(f'File: {file_name} already exists, SKIPPING')
        return
    try:
        os.makedirs('data/', exist_ok=True)
        os.makedirs(f'data/{date}', exist_ok=True)
        response = requests.get(url, stream=True)
        with open(f'data/{date}/{file_name}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded {file_name}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def download_all(date, urls):
  threads = []

  for url in urls:
    thread = threading.Thread(target=download_url, args=(date, url,))
    thread.start()
    threads.append(thread)

  for thread in threads:
    thread.join()


def delete_files_in_folder(folder_path):
  if not os.path.exists(folder_path):
    print("Folder does not exist.")
    return

  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      else:
        print(f"Skipping {filename}, as it is not a file.")
    except Exception as e:
      print(f"Failed to delete {file_path}. Reason: {e}")


def download_tiles_and_merge(start_date, end_date):
  date_list = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
  for i in date_list:
    current_date = i.strftime("%Y-%m-%d")
    target_output_tif = f'{modis_day_wise}/{current_date}__sur_refl_b01_1.tif'
    
    if os.path.exists(target_output_tif):
        file_size_bytes = os.path.getsize(target_output_tif)
        print(f"file_size_bytes: {file_size_bytes}")
        print(f"The file {target_output_tif} exists. skip.")
    else:
        print(f"The file {target_output_tif} does not exist.")
        print("start to download files from NASA server to local")
        earthaccess.login(strategy="netrc")
        results = earthaccess.search_data(short_name="MOD09GA", 
                                          cloud_hosted=True, 
                                        #   bounding_box=(-124.77, 24.52, -66.95, 49.38), # entire western US
                                          bounding_box=(-125, 43, -120, 45), # just a piece of the oregon coast
                                          temporal=(current_date, current_date))
        earthaccess.download(results, input_folder)
        print("done with downloading, start to convert HDF to geotiff..")

        convert_all_hdf_in_folder(input_folder, output_folder)
        print("done with conversion, start to merge geotiff tiles to one tif per day..")
        for band_name in bands:
            band = get_band_path_name_from_band_original_name(band_name)
            target_output_tif = f'{modis_day_wise}/{current_date}__{band}.tif'
            merge_tifs(folder_path=output_folder, 
                       target_date = current_date, 
                       output_file=target_output_tif, band_name=band)
        
    #delete_files_in_folder(input_folder)  # cleanup
    #delete_files_in_folder(output_folder)  # cleanup

    
if __name__ == "__main__":
  start_date = datetime(2018, 1, 1)
  end_date = datetime(2018, 1, 31)
  download_tiles_and_merge(start_date, end_date)


