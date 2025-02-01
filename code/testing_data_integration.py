import os
import pandas as pd
from datetime import datetime, timedelta
from snowcast_utils import homedir, work_dir, data_dir, test_start_date, test_end_date, process_dates_in_range
import sys
import numpy as np
from add_snodas_mask_column import add_snodas_mask_column

def get_water_year(date):
    if date.month >= 10:  # If the month is October or later
        return date.year + 1  # Water year starts in the following calendar year
    else:
        return date.year

def merge_all_gridmet_amsr_csv_into_one(target_date, gridmet_csv_folder, dem_all_csv, testing_all_csv, water_mask_csv):
    """
    Merge all GridMET and AMSR CSV files into one combined CSV file.

    Args:
        gridmet_csv_folder (str): The folder containing GridMET CSV files.
        dem_all_csv (str): Path to the DEM (Digital Elevation Model) CSV file.
        testing_all_csv (str): Path to save the merged CSV file.

    Returns:
        None
    """
    # List of file paths for the CSV files
    csv_files = []
    selected_date = datetime.strptime(target_date, "%Y-%m-%d")
    for file in os.listdir(gridmet_csv_folder):
        if file.endswith('_cumulative.csv') and target_date in file:
            csv_files.append(os.path.join(gridmet_csv_folder, file))

    # Initialize an empty list to store all dataframes
    all_df = None

    # Read each CSV file into separate dataframes
    for file in csv_files:
        print(f"reading {file}")
        df = pd.read_csv(file)
        # print(df.head())
        # print("df.shape:", df.shape)
        df = df.apply(pd.to_numeric, errors='coerce')
        if all_df is None:
          all_df = df
        else:
          #all_df = all_df.merge(df, on=['Latitude', 'Longitude']).drop_duplicates()
          df = df.drop(columns=['Latitude', 'Longitude'])
          all_df = pd.concat([all_df, df], axis=1)
          
        # print("all_df.head() :", all_df.head())
        # print("all_df.columns", all_df.columns)
        # print("all_df.shape: ", all_df.shape)

    unique_loc_pairs = all_df[['Latitude', 'Longitude']].drop_duplicates()
    # print("unique_loc_pairs.shape: ", unique_loc_pairs.shape)
        
    dem_df = pd.read_csv(f"{data_dir}/srtm/dem_all.csv", encoding='utf-8', index_col=False)
    #all_df = pd.merge(all_df, dem_df, on=['Latitude', 'Longitude']).drop_duplicates()
    dem_df = dem_df.drop(columns=['Latitude', 'Longitude'])
    # print("dem_df.shape: ", dem_df.shape)
    all_df = pd.concat([all_df, dem_df], axis=1)

    date = target_date
    
    #date = date.replace("-", ".")
    amsr_file = f'{data_dir}/amsr_testing/testing_ready_amsr_{date}_cumulative.csv'
    print(f"reading {amsr_file}")
    amsr_df = pd.read_csv(amsr_file, index_col=False)
    amsr_df.rename(columns={'gridmet_lat': 'Latitude', 'gridmet_lon': 'Longitude'}, inplace=True)
    # print(amsr_df.head())
    # print("amsr_df.shape = ", amsr_df.shape)
    amsr_df = amsr_df.drop(columns=['Latitude', 'Longitude'])
    all_df = pd.concat([all_df, amsr_df], axis=1)
    
    fsca_df = pd.read_csv(f'{data_dir}/fsca/final_output/{target_date}_output.csv')
    # print(fsca_df.head())
    # print("fsca_df.shape: ", fsca_df.shape)
    fsca_df = fsca_df.drop(columns=['Latitude', 'Longitude'])
    all_df = pd.concat([all_df, fsca_df], axis=1)

    water_mask_df = pd.read_csv(water_mask_csv)
    water_mask_df = water_mask_df.drop(columns=["date", 'Latitude', 'Longitude'])
    all_df = pd.concat([all_df, water_mask_df], axis=1)
    
    # print("all columns: ", all_df.columns)
    # add water year
    all_df["water_year"] = get_water_year(selected_date)
    
    all_df.rename(columns={'date_x': 'date'}, inplace=True)
    
    # log10 all the cumulative columns
    # Get columns with "cumulative" in their names
    for col in all_df.columns:
        print("Checking ", col)
        if "cumulative" in col:
	        # Apply log10 transformation to selected columns
            all_df[col] = np.log10(all_df[col] + 0.1)  # Adding 1 to avoid log(0)
            print(f"converted {col} to log10")
    
    # Remove the duplicated date column
    all_df = all_df.loc[:, ~all_df.columns.duplicated()]

    # Save the merged dataframe to a new CSV file
    all_df.to_csv(testing_all_csv, index=False)
    print(f"All input CSV files are merged to {testing_all_csv}")
    print(all_df.columns)
    print("all_df.shape = ", all_df.shape)


    # Call the function with appropriate paths
    # training_csv = f"{work_dir}/all_points_final_merged_training.csv"
    # new_training_csv = f"{work_dir}/all_points_final_merged_training_snodas_mask.csv"
    model_path = f"/home/chetana/models//SNODASDNNHole_e10_nTrue_20253101055710.model"
    add_snodas_mask_column(
        model_path, 
        testing_all_csv, 
        f"{testing_all_csv}_snodas_mask.csv",
        lat_col_name = "Latitude", 
        lon_col_name = "Longitude", 
        date_col_name = "date"
    )

def merge_callback(current_date):
    # Prepare the cumulative history CSVs for the current date
    print(">>>>>\nGetting gridmet for day", current_date.strftime("%Y-%m-%d"))
    current_date_str = current_date.strftime("%Y-%m-%d")
    test_year = "2024"

    merge_all_gridmet_amsr_csv_into_one(
        current_date_str, 
        f"{work_dir}/testing_output/",
        f"{data_dir}/srtm/dem_all.csv",
        f"{work_dir}/testing_all_ready_{current_date_str}.csv",
        f"{data_dir}/water_mask/final_output/{test_year}_output.csv"
    )

if __name__ == "__main__":
    process_dates_in_range(
        start_date = test_start_date,
        end_date = test_end_date,
        days_look_back=7,
        callback = merge_callback,
    )

