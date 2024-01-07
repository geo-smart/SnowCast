"""
This script performs the following operations:
1. Reads multiple CSV files into Dask DataFrames with specified chunk sizes and compression.
2. Repartitions the DataFrames for optimized processing.
3. Merges the DataFrames based on specified columns.
4. Saves the merged DataFrame to a CSV file in chunks.
5. Reads the merged DataFrame, removes duplicate rows, and saves the cleaned DataFrame to a new CSV file.

Attributes:
    working_dir (str): The directory where the CSV files are located.
    chunk_size (str): The chunk size used for reading and processing the CSV files.

Functions:
    main(): The main function that executes the data processing operations and saves the results.
"""

import dask.dataframe as dd
import os
from snowcast_utils import work_dir, homedir
import pandas as pd

working_dir = work_dir
final_output_name = "final_merged_data_3yrs_all_active_stations_v1.csv"
chunk_size = '10MB'  # You can adjust this chunk size based on your hardware and data size


def merge_all_data_together():
    amsr_file = f'{working_dir}/all_snotel_cdec_stations_active_in_westus.csv_amsr_dask.csv'
    snotel_file = f'{working_dir}/all_snotel_cdec_stations_active_in_westus.csv_swe_restored_dask_all_vars.csv'
    gridmet_file = f'{working_dir}/training_all_active_snotel_station_list_elevation.csv_gridmet.csv'
    terrain_file = f'{working_dir}/training_all_active_snotel_station_list_elevation.csv_terrain_4km_grid_shift.csv'
    fsca_file = f'{homedir}/fsca/fsca_final_training_all.csv'
    final_final_output_file = f'{work_dir}/{final_output_name}'
    
    if os.path.exists(final_final_output_file):
      print(f"The file '{final_final_output_file}' exists. Skipping")
      return final_final_output_file
      
    
    # Read the CSV files with a smaller chunk size and compression
    amsr = dd.read_csv(amsr_file, blocksize=chunk_size)
    print("amsr.columns = ", amsr.columns)
    snotel = dd.read_csv(snotel_file, blocksize=chunk_size)
    print("snotel.columns = ", snotel.columns)
#     gridmet = dd.read_csv(f'{working_dir}/gridmet_climatology/training_ready_gridmet.csv', blocksize=chunk_size)
    gridmet = dd.read_csv(gridmet_file, blocksize=chunk_size)
    gridmet = gridmet.drop(columns=["Unnamed: 0"])
    print("gridmet.columns = ", gridmet.columns)
    terrain = dd.read_csv(terrain_file, blocksize=chunk_size)
    terrain = terrain.rename(columns={
      "latitude": "lat", 
      "longitude": "lon"
    })
    terrain = terrain[["stationTriplet", "elevation", "lat", "lon", 'Elevation', 'Slope', 'Aspect', 'Curvature', 'Northness', 'Eastness']]
    print("terrain.columns = ", terrain.columns)
    snowcover = dd.read_csv(fsca_file, blocksize=chunk_size)
    snowcover = snowcover.rename(columns={
      "latitude": "lat", 
      "longitude": "lon"
    })
    print("snowcover.columns = ", snowcover.columns)

    # Repartition DataFrames for optimized processing
    amsr = amsr.repartition(partition_size=chunk_size)
    snotel = snotel.repartition(partition_size=chunk_size)
    gridmet = gridmet.repartition(partition_size=chunk_size)
    gridmet = gridmet.rename(columns={'day': 'date'})
    terrain = terrain.repartition(partition_size=chunk_size)
    snow_cover = snowcover.repartition(partition_size=chunk_size)
    print("all the dataframes are partitioned")

    # Merge DataFrames based on specified columns
    print("start to merge amsr and snotel")
    merged_df = dd.merge(amsr, snotel, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_snotel.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    print("start to merge gridmet")
    merged_df = dd.merge(merged_df, gridmet, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_gridmet.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    print("start to merge terrain")
    merged_df = dd.merge(merged_df, terrain, on=['lat', 'lon'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_terrain.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    print("start to merge snowcover")
    merged_df = dd.merge(merged_df, snow_cover, on=['lat', 'lon', 'date'], how='outer')
    merged_df = merged_df.drop_duplicates(keep='first')
    output_file = os.path.join(working_dir, f"{final_output_name}_snow_cover.csv")
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f"intermediate file saved to {output_file}")
    
    # Save the merged DataFrame to a CSV file in chunks
    output_file = os.path.join(working_dir, final_output_name)
    merged_df.to_csv(output_file, single_file=True, index=False)
    print(f'Merge completed. {output_file}')

    # Read the merged DataFrame, remove duplicate rows, and save the cleaned DataFrame to a new CSV file
    df = dd.read_csv(f'{work_dir}/{final_output_name}')
    df = df.drop_duplicates(keep='first')
    df.to_csv(f'{work_dir}/{final_output_name}', single_file=True, index=False)
    print('Data cleaning completed.')
    return final_final_output_file

  
def sort_training_data(input_training_csv, sorted_training_csv):
    # Read Dask DataFrame from CSV with increased blocksize and assuming missing data
    ddf = dd.read_csv(input_training_csv, assume_missing=True, blocksize='10MB')

    # Persist the Dask DataFrame in memory
    ddf = ddf.persist()

    # Sort Dask DataFrame by three columns: date, lat, and Lon
    sorted_ddf = ddf.sort_values(by=['date', 'lat', 'lon'])

    # Save the sorted Dask DataFrame to a new CSV file
    sorted_ddf.to_csv(sorted_training_csv, index=False, single_file=True)
    print(f"sorted training data is saved to {sorted_training_csv}")
  
if __name__ == "__main__":
    merge_all_data_together()
    final_final_output_file = f'{work_dir}/{final_output_name}'
    sort_training_data(final_final_output_file, f'{work_dir}/{final_output_name}_sorted.csv')
