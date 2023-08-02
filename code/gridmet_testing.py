#############################################
# Process Name: gridmet_station_only
# Person Assigned: Gokul Prathin A
# Last Changes On: 1st July 2023
#############################################

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import urllib.request
from datetime import datetime, timedelta, date

def get_current_year():
    now = datetime.now()
    current_year = now.year
    return current_year

year_list = [get_current_year()]

def remove_files_in_folder(folder_path):
    # Get a list of files in the folder
    files = os.listdir(folder_path)

    # Loop through the files and remove them
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def download_file(url, target_file_path, variable):
    try:
        with urllib.request.urlopen(url) as response:
            print(f"Downloading {url}")
            file_content = response.read()
        save_path = target_file_path
        with open(save_path, 'wb') as file:
            file.write(file_content)
        print(f"File downloaded successfully and saved as: {save_path}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {str(e)}")


def download_gridmet_of_specific_variables():
    # make a directory to store the downloaded files
    

    base_metadata_url = "http://www.northwestknowledge.net/metdata/data/"
    variables_list = ['tmmn', 'tmmx', 'pr', 'vpd', 'etr', 'rmax', 'rmin', 'vs']

    for var in variables_list:
        for y in year_list:
            download_link = base_metadata_url + var + '_' + '%s' % y + '.nc'
            target_file_path = os.path.join(folder_name, var + '_' + '%s' % y + '.nc')
            if not os.path.exists(target_file_path):
                download_file(download_link, target_file_path, var)
            else:
                print(f"File {target_file_path} exists")

folder_name = '/home/chetana/gridmet_test_run/gridmet_climatology'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
remove_files_in_folder(folder_name)
download_gridmet_of_specific_variables()


