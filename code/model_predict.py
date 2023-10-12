import joblib
import pandas as pd
import netCDF4 as nc
from datetime import timedelta, datetime

model = joblib.load('/home/chetana/gridmet_test_run/model_creation_et.pkl')
new_data = pd.read_csv("/home/chetana/gridmet_test_run/testing_all_ready.csv")
reference_nc_file = nc.Dataset('/home/chetana/gridmet_test_run/gridmet_climatology/etr_2023.nc')

reference_date = datetime(1900, 1, 1)
day = reference_nc_file.variables['day'][:]

day_value = day[-1]
day_value = 44998
print('current day count:', day_value)

result_date = reference_date + timedelta(days=day_value)
new_data['date'] = result_date.strftime("%Y-%m-%d")

new_data['date'] = pd.to_datetime(new_data['date'])
new_data['year'] = new_data['date'].dt.year
new_data['month'] = new_data['date'].dt.month
new_data['day'] = new_data['date'].dt.day
new_data['day_of_week'] = new_data['date'].dt.dayofweek

new_data.drop(['swe_change', 'snow_depth_change'], axis=1, inplace=True, errors='ignore')
new_data.drop('date', axis=1, inplace=True)
new_data.replace('--', pd.NA, inplace=True)

new_data.rename(columns={'Elevation': 'elevation', 'Slope': 'slope',
                         'Aspect': 'aspect', 'Curvature': 'curvature',
                         'Northness': 'northness', 'Eastness': 'eastness',
                         'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# Handle missing values by replacing with the mean of each column
numerical_columns = ['lat', 'lon', 'vpd', 'vs', 'pr', 'rmax', 'etr', 'tmmn', 'tmmx', 'rmin', 'elevation', 'slope',
                     'aspect', 'curvature', 'northness', 'eastness']
new_data[numerical_columns] = new_data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Calculate the mean of each column
mean_values = new_data[numerical_columns].mean()

columns_to_delete = [0, 4, 6, 8, 10, 12, 14, 16, 18, 19]
new_data.drop(new_data.columns[columns_to_delete], axis=1, inplace=True)

# Fill missing data with mean values
#new_data[numerical_columns] = new_data[numerical_columns].fillna(mean_values)
new_data.dropna(inplace=True)

# ['date', 'lat', 'lon', 'etr', 'pr', 'rmax',
#                     'rmin', 'tmmn', 'tmmx', 'vpd', 'vs', 
#                     'elevation',
#                     'slope', 'curvature', 'aspect', 'eastness',
#                     'northness', 'Snow Water Equivalent (in) Start of Day Values']

desired_order = ['lat', 'lon', 'etr', 'pr', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd', 'vs',
       'elevation', 'slope', 'curvature', 'aspect', 'eastness', 'northness',
       'year', 'month', 'day', 'day_of_week']

# Reindex the DataFrame with the desired order of columns
new_data = new_data.reindex(columns=desired_order)

new_predictions = model.predict(new_data)

new_data['predicted_swe'] = new_predictions

new_data.to_csv('/home/chetana/gridmet_test_run/test_data_prediected.csv', index=False)

print("prediction successfully done")
