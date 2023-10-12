import os
import pandas as pd
import netCDF4 as nc
import csv
from datetime import datetime


gridmet_var_mapping = {
  "etr": "potential_evapotranspiration",
  "pr":"precipitation_amount",
  "rmax":"relative_humidity",
  "rmin":"relative_humidity",
  "tmmn":"air_temperature",
  "tmmx":"air_temperature",
  "vpd":"mean_vapor_pressure_deficit",
  "vs":"wind_speed",
}


dem_csv = "/home/chetana/gridmet_test_run/dem_all.csv"


def get_current_year():
    now = datetime.now()
    current_year = now.year
    return current_year


def get_file_name_from_path(file_path):
    # Get the file name from the file path
    file_name = os.path.basename(file_path)
    return file_name

def get_var_from_file_name(file_name):
    # Assuming the file name format is "tmmm_year.csv"
    var_name = str(file_name.split('_')[0])
    return var_name

def get_coordinates_of_template_tif():
  	# Load the CSV file and extract coordinates
    coordinates = []
    df = pd.read_csv(dem_csv)
    for index, row in df.iterrows():
        # Process each row here
        lon, lat = float(row["Longitude"]), float(row["Latitude"])
        coordinates.append((lon, lat))
    return coordinates

def find_nearest_index(array, value):
    # Find the index of the element in the array that is closest to the given value
    return (abs(array - value)).argmin()

def get_nc_csv_by_coords_and_variable(nc_file, coordinates, var_name):
    coordinates = get_coordinates_of_template_tif()
    # get the netcdf file and generate the csv file for every coordinate in the dem_template.csv
    new_lat_data = []
    new_lon_data = []
    new_var_data = []
    # Read the NetCDF file
    with nc.Dataset(nc_file) as nc_file:
      # Get a list of all variables in the NetCDF file
      variables = nc_file.variables.keys()

      # Print the variables and their shapes
      for variable in variables:
        shape = nc_file.variables[variable].shape
        print(f"Variable: {variable}, Shape: {shape}")
      
      # Get the values at each coordinate using rasterio's sample function
      latitudes = nc_file.variables['lat'][:]
      longitudes = nc_file.variables['lon'][:]
      day = nc_file.variables['day'][:]
      long_var_name = gridmet_var_mapping[var_name]
      print("long var name: ", long_var_name)
      var_col = nc_file.variables[long_var_name][:]
      
      print(f"latitudes shape: {latitudes.shape}")
      print(f"longitudes shape: {longitudes.shape}")
      print(f"day shape: {day.shape}")
      print(f"val col shape: {var_col.shape}")
      
      day_index = day[day.shape[0]-1]
      day_index = 44998
      print('day_index:', day_index)
      
      for coord in coordinates:
        lon, lat = coord
        new_lat_data.append(lat)
        new_lon_data.append(lon)
        # Access the variables in the NetCDF file
        # Find the nearest indices for the given coordinates
        lon_index = find_nearest_index(longitudes, lon)
        lat_index = find_nearest_index(latitudes, lat)
        #day_index = find_nearest_index(day, day[day.shape[0]-1])
        #print(f"last day: {day_index}")

        # Get the value at the specified coordinates
        the_value = var_col[day.shape[0]-1, lat_index, lon_index]  # Assuming data_variable is a 3D variable (time, lat, lon)
        if the_value == "--":
          the_value = -9999
        new_var_data.append(the_value)
        #print(f"lon - {lon} lat - {lat} lon-index {lon_index} lat-index {lat_index} day-index {day_index} value - {the_value}")
    # Create the DataFrame
    data = { 'Latitude': new_lat_data, 'Longitude': new_lon_data, var_name: new_var_data}
    df = pd.DataFrame(data)
    return df

def turn_gridmet_nc_to_csv(folder_path, dem_all_csv, testing_all_csv):
    coordinates = get_coordinates_of_template_tif()
    current_year = get_current_year()
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            var_name = get_var_from_file_name(file_name)
            print("Variable name:", var_name)
            res_csv = f"/home/chetana/gridmet_test_run/testing_output/{str(current_year)}_{var_name}.csv"
            if os.path.exists(res_csv):
                os.remove(res_csv)
                print(f"remove old {res_csv}")
            
            if str(current_year) in file_name :
                # Perform operations on each file here
                netcdf_file_path = os.path.join(root, file_name)
                print("Processing file:", netcdf_file_path)
                file_name = get_file_name_from_path(netcdf_file_path)
                print("File Name:", file_name)

                df = get_nc_csv_by_coords_and_variable(netcdf_file_path, coordinates, var_name)
                df.to_csv(res_csv)
            
def merge_all_gridmet_csv_into_one(gridmet_csv_folder, dem_all_csv, testing_all_csv):
    # List of file paths for the CSV files
    csv_files = []
    for file in os.listdir(gridmet_csv_folder):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(gridmet_csv_folder, file))

    # Initialize an empty list to store all dataframes
    dfs = []

    # Read each CSV file into separate dataframes
    for file in csv_files:
        df = pd.read_csv(file, encoding='utf-8', index_col=False)
        dfs.append(df)

    dem_df = pd.read_csv(dem_all_csv, encoding='utf-8', index_col=False)
    dfs.append(dem_df)
    
    # Merge the dataframes based on the latitude and longitude columns
    merged_df = dfs[0]  # Start with the first dataframe
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on=['Latitude', 'Longitude'])

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(testing_all_csv, index=False)
    print(f"All input csv files are merged to {testing_all_csv}")
    print(merged_df.head())

    

if __name__ == "__main__":
    # Replace with the actual path to your folder
    gridmet_csv_folder = "/home/chetana/gridmet_test_run/gridmet_climatology/"
    #turn_gridmet_nc_to_csv(gridmet_csv_folder)
    merge_all_gridmet_csv_into_one("/home/chetana/gridmet_test_run/testing_output/",
                                  "/home/chetana/gridmet_test_run/dem_all.csv",
                                  "/home/chetana/gridmet_test_run/testing_all_ready.csv")


