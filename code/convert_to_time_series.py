import pandas as pd
import os
from snowcast_utils import work_dir
import shutil
import numpy as np
import dask.dataframe as dd

# Set Pandas options to display all columns
pd.set_option('display.max_columns', None)



def array_describe(arr):
    """
    Calculate descriptive statistics for a given NumPy array.

    Args:
        arr (numpy.ndarray): The input array for which statistics are calculated.

    Returns:
        dict: A dictionary containing descriptive statistics such as Mean, Median, Standard Deviation, Variance, Minimum, Maximum, and Sum.
    """
    stats = {
        'Mean': np.mean(arr),
        'Median': np.median(arr),
        'Standard Deviation': np.std(arr),
        'Variance': np.var(arr),
        'Minimum': np.min(arr),
        'Maximum': np.max(arr),
        'Sum': np.sum(arr),
    }
    return stats

def interpolate_missing_inplace(df, column_name, degree=3):
    x = df.index
    y = df[column_name]

    # Create a mask for missing values
    if column_name == "SWE":
      mask = (y > 240) | y.isnull()
    elif column_name == "fsca":
      mask = (y > 100) | y.isnull()
    else:
      mask = y.isnull()

    # Check if all elements in the mask array are True
    all_true = np.all(mask)

    if all_true:
      df[column_name] = 0
    else:
      # Perform interpolation
      new_y = np.interp(x, x[~mask], y[~mask])
      # Replace missing values with interpolated values
      df[column_name] = new_y

    if np.any(df[column_name].isnull()):
      raise ValueError("Single group: shouldn't have null values here")
        
    return df

def convert_to_time_series(input_csv, output_csv):
    """
    Convert the data from the ready CSV file into a time series format.

    This function reads the cleaned CSV file, sorts the data, fills in missing values using polynomial interpolation, and creates a time series dataset for specific columns. The resulting time series data is saved to a new CSV file.
    """
    
    columns_to_be_time_series = ["SWE", 
                                 'air_temperature_tmmn',
                                 'potential_evapotranspiration', 
                                 'mean_vapor_pressure_deficit',
                                 'relative_humidity_rmax', 
                                 'relative_humidity_rmin',
                                 'precipitation_amount', 
                                 'air_temperature_tmmx', 
                                 'wind_speed',
                                 'fsca']

    # Read the cleaned ready CSV
    df = pd.read_csv(input_csv)
    df.sort_values(by=['lat', 'lon', 'date'], inplace=True)
    print("All current columns: ", df.columns)
    
    # rename all columns to unified names
    #     ['date', 'lat', 'lon', 'SWE', 'Flag', 'swe_value', 'cell_id',
# 'station_id', 'etr', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd', 'vs',
# 'elevation', 'slope', 'curvature', 'aspect', 'eastness', 'northness',
# 'fsca']
    df.rename(columns={'vpd': 'mean_vapor_pressure_deficit',
                         'vs': 'wind_speed', 
                         'pr': 'precipitation_amount', 
                         'etr': 'potential_evapotranspiration',
                         'tmmn': 'air_temperature_tmmn',
                         'tmmx': 'air_temperature_tmmx',
                         'rmin': 'relative_humidity_rmin',
                         'rmax': 'relative_humidity_rmax',
                         'AMSR_SWE': 'SWE',
                        }, inplace=True)
    
    filled_csv = f"{output_csv}_gap_filled.csv"
    if os.path.exists(filled_csv):
        print(f"{filled_csv} already exists, skipping")
        filled_data = pd.read_csv(filled_csv)
    else:
        # Function to perform polynomial interpolation and fill in missing values
        def process_group_filling_value(group):
          # Sort the group by 'date'
          group = group.sort_values(by='date')
      
          for column_name in columns_to_be_time_series:
            group = interpolate_missing_inplace(group, column_name)
          # Return the processed group
          return group
        # Group the data by 'lat' and 'lon' and apply interpolation for each column
        print("Start to fill in the missing values")
        grouped = df.groupby(['lat', 'lon'])
        filled_data = grouped.apply(process_group_filling_value).reset_index(drop=True)
    

        if any(filled_data['fsca'] > 100):
          raise ValueError("Error: shouldn't have SWE>240 at this point")

        filled_data.to_csv(filled_csv, index=False)
        
        print(f"New filled values csv is saved to {filled_csv}")
    
    if os.path.exists(output_csv):
        print(f"{output_csv} already exists, skipping")
    else:
        df = filled_data
        # Create a new DataFrame to store the time series data for each location
        print("Start to create the training csv with previous 7 days columns")
        result = pd.DataFrame()

        # Define the number of days to consider (7 days in this case)
        num_days = 7

        grouped = df.groupby(['lat', 'lon'])
        
        def process_group_time_series(group, num_days):
          group = group.sort_values(by='date')
          for day in range(1, num_days + 1):
            for target_col in columns_to_be_time_series:
              new_column_name = f'{target_col}_{day}'
              group[new_column_name] = group[target_col].shift(day)
              
          return group
        
        result = grouped.apply(lambda group: process_group_time_series(group, num_days)).reset_index(drop=True)
        result.fillna(0, inplace=True)
        
        result.to_csv(output_csv, index=False)
        print(f"New data is saved to {output_csv}")
        shutil.copy(output_csv, backup_time_series_csv_path)
        print(f"File is backed up to {backup_time_series_csv_path}")

def add_cumulative_columns(input_csv, output_csv, force=False):
    """
    Add cumulative columns to the time series dataset.

    This function reads the time series CSV file created by `convert_to_time_series`, calculates cumulative values for specific columns, and saves the data to a new CSV file.
    """
    
    columns_to_be_cumulated = [
      "SWE",
      'air_temperature_tmmn',
      'potential_evapotranspiration', 
      'mean_vapor_pressure_deficit',
      'relative_humidity_rmax', 
      'relative_humidity_rmin',
      'precipitation_amount', 
      'air_temperature_tmmx', 
      'wind_speed',
      'fsca'
    ]

    # Read the time series CSV (ensure it was created using `convert_to_time_series` function)
    # directly read from original file
    df = pd.read_csv(input_csv)
    print("the column statistics from time series before cumulative: ", df.describe())
    
    df['date'] = pd.to_datetime(df['date'])
    
    unique_years = df['date'].dt.year.unique()
    print("This is our unique years", unique_years)
    #current_df['fSCA'] = current_df['fSCA'].fillna(0)
    
    # only start from the water year 10-01
    # Filter rows based on the date range (2019 to 2022)
    start_date = pd.to_datetime('2018-10-01')
    end_date = pd.to_datetime('2021-09-30')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    print("how many rows are left in the three water years?", df.describe())
    df.to_csv(f"{current_ready_csv_path}.test_check.csv")

    # Define a function to calculate the water year
    def calculate_water_year(date):
        year = date.year
        if date.month >= 10:  # Water year starts in October
            return year + 1
        else:
            return year
    
    # every water year starts at Oct 1, and ends at Sep 30. 
    df['water_year'] = df['date'].apply(calculate_water_year)
    
    # Group the DataFrame by 'lat' and 'lon'
    grouped = df.groupby(['lat', 'lon', 'water_year'], group_keys=False)
    print("how many groups? ", grouped)
    
    grouped = df.groupby(['lat', 'lon', 'water_year'], group_keys=False)
    for column in columns_to_be_cumulated:
        df[f'cumulative_{column}'] = grouped.apply(lambda group: group.sort_values('date')[column].cumsum())

    print("This is the dataframe after cumulative columns are added")
    print(df["cumulative_fsca"].describe())
    
    df.to_csv(output_csv, index=False)
    
    print(f"All the cumulative variables are added successfully! {target_time_series_cumulative_csv_path}")
    print("double check the swe_value statistics:", df["swe_value"].describe())

def clean_non_swe_rows(current_ready_csv_path, cleaned_csv_path):
    # Read Dask DataFrame from CSV
    dask_df = dd.read_csv(current_ready_csv_path)

    # Remove rows where 'swe_value' is empty
    dask_df_filtered = dask_df.dropna(subset=['swe_value'])

    # Save the result to a new CSV file
    dask_df_filtered.to_csv(cleaned_csv_path, index=False, single_file=True)
    print("dask_df_filtered.shape = ", dask_df_filtered.shape)
    print(f"The filtered csv with no swe values is saved to {cleaned_csv_path}")

if __name__ == "__main__":
    # Define file paths for various CSV files
    # current_ready_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3.csv'
    current_ready_csv_path = f'{work_dir}/final_merged_data_3yrs_all_active_stations_v1.csv_sorted.csv'
    cleaned_csv_path = f"{current_ready_csv_path}_cleaned_nodata.csv"
    target_time_series_csv_path = f'{cleaned_csv_path}_time_series_v1.csv'
    backup_time_series_csv_path = f'{cleaned_csv_path}_time_series_v1_bak.csv'
    # target_time_series_cumulative_csv_path = f'{work_dir}/final_merged_data_3yrs_cleaned_v3_time_series_cumulative_v1.csv'
    target_time_series_cumulative_csv_path = f'{cleaned_csv_path}_time_series_cumulative_v1.csv'
    
    
    # remove the empty swe_value rows first
    clean_non_swe_rows(current_ready_csv_path, cleaned_csv_path)
  
    # Uncomment this line to execute the 'convert_to_time_series' function
    convert_to_time_series(cleaned_csv_path, target_time_series_csv_path)

    # Uncomment this line to execute the 'add_cumulative_columns' function
    add_cumulative_columns(target_time_series_csv_path, target_time_series_cumulative_csv_path, force=True)

