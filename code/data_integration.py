import dask.dataframe as dd

# Define the file paths of the three CSV files
file_path1 = '/home/chetana/gridmet_test_run/climatology_data.csv'
file_path2 = '/home/chetana/gridmet_test_run/training_ready_snotel_data.csv'
file_path3 = '/home/chetana/gridmet_test_run/training_ready_terrain.csv'

# Read each CSV file into a Dask DataFrame
df1 = dd.read_csv(file_path1)
df2 = dd.read_csv(file_path2)
df3 = dd.read_csv(file_path3)

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

