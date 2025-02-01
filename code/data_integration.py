"""
This script reads three CSV files into Dask DataFrames, performs data type conversion,
and merges them into a final Dask DataFrame. The merged data is then saved to a CSV file.

The three CSV files include climatology data, training-ready SNOTEL data, and training-ready
terrain data, each with latitude ('lat'), longitude ('lon'), and date ('date') columns.

Attributes:
    file_path1 (str): File path of the climatology data CSV file.
    file_path2 (str): File path of the training-ready SNOTEL data CSV file.
    file_path3 (str): File path of the training-ready terrain data CSV file.

Functions:
    small_function: Reads, processes, and merges the CSV files and saves the result to a CSV file.
"""

import dask.dataframe as dd
from snowcast_utils import homedir, work_dir, train_start_date, train_end_date, fsca_dir
import os

chunk_size = '10MB'  # You can adjust this chunk size based on your hardware and data size
# Define the file paths of the three CSV files
file_path1 = '/home/chetana/gridmet_test_run/climatology_data.csv'
file_path2 = '/home/chetana/gridmet_test_run/training_ready_snotel_data.csv'
file_path3 = '/home/chetana/gridmet_test_run/training_ready_terrain.csv'

def small_function():
    """
    Reads each CSV file into a Dask DataFrame, performs data type conversion for latitude and longitude,
    merges the DataFrames based on specific columns, and saves the merged Dask DataFrame to a CSV file.

    Args:
        None

    Returns:
        None
    """
    # Read each CSV file into a Dask DataFrame
    df1 = dd.read_csv(file_path1)
    df2 = dd.read_csv(file_path2)
    df3 = dd.read_csv(file_path3)

    # Perform data type conversion for latitude and longitude columns
    df1['lat'] = df1['lat'].astype(float)
    df1['lon'] = df1['lon'].astype(float)
    df2['lat'] = df2['lat'].astype(float)
    df2['lon'] = df2['lon'].astype(float)
    df3['lat'] = df3['lat'].astype(float)
    df3['lon'] = df3['lon'].astype(float)

    # Merge the first two DataFrames based on 'lat', 'lon', and 'date'
    merged_df1 = dd.merge(df1, df2, left_on=['lat', 'lon', 'date'], right_on=['lat', 'lon', 'Date'])

    # Merge the third DataFrame based on 'lat' and 'lon'
    merged_df2 = dd.merge(merged_df1, df3, on=['lat', 'lon'])

    # Save the merged Dask DataFrame directly to a CSV file
    merged_df2.to_csv('/home/chetana/gridmet_test_run/model_training_data.csv', index=False, single_file=True)

# Uncomment the line below to execute the function
# small_function()

def merge_all_files_for_training_points(training_points_csv, final_final_output_file):
    training_points_file_name = os.path.basename(training_points_csv)

    terrain_file = f"{work_dir}/{training_points_file_name}_dem.csv"
    gridmet_file = f"{work_dir}/{training_points_file_name}_gridmet_training.csv"
    amsr_file = f"{work_dir}/{training_points_file_name}_amsr_training.csv"
    fsca_file = f"{fsca_dir}/{training_points_file_name}_fsca_training.csv"
    
    if os.path.exists(final_final_output_file):
      print(f"The file '{final_final_output_file}' exists. Skipping")
      return final_final_output_file
    
    # Read the CSV files with a smaller chunk size and compression
    amsr = dd.read_csv(amsr_file, blocksize=chunk_size)
    print("amsr.columns = ", amsr.columns)
    # ground_truth = dd.read_csv(all_station_obs_file, blocksize=chunk_size)
    # print("ground_truth.columns = ", ground_truth.columns)
#     gridmet = dd.read_csv(f'{working_dir}/gridmet_climatology/training_ready_gridmet.csv', blocksize=chunk_size)
    gridmet = dd.read_csv(gridmet_file, blocksize=chunk_size)
    gridmet = gridmet.drop(columns=["Unnamed: 0"])
    print("gridmet.columns = ", gridmet.columns)
    terrain = dd.read_csv(terrain_file, blocksize=chunk_size)
    terrain = terrain.rename(columns={
      "latitude": "lat", 
      "longitude": "lon"
    })
    terrain = terrain[["lat", "lon", 'Elevation', 'Slope', 'Aspect', 'Curvature', 'Northness', 'Eastness']]
    print("terrain.columns = ", terrain.columns)
    snowcover = dd.read_csv(fsca_file, blocksize=chunk_size)
    snowcover = snowcover.rename(columns={
      "latitude": "lat", 
      "longitude": "lon"
    })
    print("snowcover.columns = ", snowcover.columns)

    # Repartition DataFrames for optimized processing
    amsr = amsr.repartition(partition_size=chunk_size)
    # ground_truth = ground_truth.repartition(partition_size=chunk_size)
    gridmet = gridmet.repartition(partition_size=chunk_size)
    gridmet = gridmet.rename(columns={'day': 'date'})
    terrain = terrain.repartition(partition_size=chunk_size)
    snow_cover = snowcover.repartition(partition_size=chunk_size)
    print("all the dataframes are partitioned")

    # Merge DataFrames based on specified columns
    # print("start to merge amsr and ground_truth")
    # merged_df = dd.merge(amsr, ground_truth, on=['lat', 'lon', 'date'], how='outer')
    # merged_df = merged_df.drop_duplicates(keep='first')
    # output_file = os.path.join(working_dir, f"{final_output_name}_ground_truth.csv")
    # merged_df.to_csv(output_file, single_file=True, index=False)
    # print(f"intermediate file saved to {output_file}")
    
    print("start to merge amsr and gridmet")
    merged_df = dd.merge(amsr, gridmet, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    # output_file = os.path.join(work_dir, f"{final_output_name}_gridmet.csv")
    # merged_df.to_csv(output_file, single_file=True, index=False)
    # print(f"intermediate file saved to {output_file}")
    
    print("start to merge terrain")
    merged_df = dd.merge(merged_df, terrain, on=['lat', 'lon'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    # output_file = os.path.join(working_dir, f"{final_output_name}_terrain.csv")
    # merged_df.to_csv(output_file, single_file=True, index=False)
    # print(f"intermediate file saved to {output_file}")
    
    print("start to merge snowcover")
    merged_df = dd.merge(merged_df, snow_cover, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    # output_file = os.path.join(working_dir, f"{final_output_name}_snow_cover.csv")
    # merged_df.to_csv(output_file, single_file=True, index=False)
    # print(f"intermediate file saved to {output_file}")
    
    # Save the merged DataFrame to a CSV file in chunks
    # output_file = os.path.join(working_dir, final_output_name)
    merged_df.to_csv(final_final_output_file, single_file=True, index=False)
    print(f'Merge completed. {final_final_output_file}')

def cleanup_dataframe(training_points_csv, final_final_output_file):
    """
    Read the merged DataFrame, remove duplicate rows, and save the cleaned DataFrame to a new CSV file
    """
    dtype = {'station_name': 'object'}  # 'object' dtype represents strings
    df = dd.read_csv(final_final_output_file, dtype=dtype)
    df = df.drop_duplicates(keep='first')
    df.to_csv(final_final_output_file, single_file=True, index=False)
    print('Data cleaning completed.')
    return final_final_output_file

  
def sort_training_data(input_training_csv, sorted_training_csv):
    # Read Dask DataFrame from CSV with increased blocksize and assuming missing data
    dtype = {'station_name': 'object'}  # 'object' dtype represents strings
    ddf = dd.read_csv(input_training_csv, assume_missing=True, blocksize='10MB', dtype=dtype)

    # Persist the Dask DataFrame in memory
    ddf = ddf.persist()

    # Sort Dask DataFrame by three columns: date, lat, and Lon
    sorted_ddf = ddf.sort_values(by=['date', 'lat', 'lon'])

    # Save the sorted Dask DataFrame to a new CSV file
    sorted_ddf.to_csv(sorted_training_csv, index=False, single_file=True)
    print(f"sorted training data is saved to {sorted_training_csv}")
  

if __name__ == "__main__":
    training_points_csv = f"{work_dir}/salt_pepper_points_for_training.csv"
    final_final_output_file = f"{training_points_csv}_all_training.csv"
    merge_all_files_for_training_points(training_points_csv, final_final_output_file)
    cleanup_dataframe(training_points_csv, final_final_output_file)
    sort_training_data(final_final_output_file, f"{final_final_output_file}_sorted.csv")



