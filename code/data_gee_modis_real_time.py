

# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernal

from all_dependencies import *

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate() # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Initialize()

# read the grid geometry file
homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"

org_name = 'modis'
product_name = f'MODIS/006/MOD10A1'
var_name = 'NDSI'
column_name = 'mod10a1_ndsi'
start_date = '2022-03-07'
end_date = '2022-03-13'

final_csv_file = f"{homedir}/Documents/GitHub/SnowCast/data/sat_testing/{org_name}/{column_name}_{start_date}_{end_date}.csv"
print(f"Results will be saved to {final_csv_file}")

if os.path.exists(final_csv_file):
     print("exists exiting..")
     exit()

station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)

all_cell_df = pd.DataFrame(columns = ['date', column_name, 'cell_id', 'latitude', 'longitude'])

for ind in station_cell_mapper_df.index:
    
    try:
      
  	  print(station_cell_mapper_df['station_id'][ind], station_cell_mapper_df['cell_id'][ind])
  	  current_cell_id = station_cell_mapper_df['cell_id'][ind]
  	  print("collecting ", current_cell_id)

  	  longitude = station_cell_mapper_df['lon'][ind]
  	  latitude = station_cell_mapper_df['lat'][ind]

  	  # identify a 500 meter buffer around our Point Of Interest (POI)
  	  poi = ee.Geometry.Point(longitude, latitude).buffer(30)

  	  def poi_mean(img):
  	      reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30)
  	      mean = reducer.get(var_name)
  	      return img.set('date', img.date().format()).set(column_name,mean)
        
  	  viirs1 = ee.ImageCollection(product_name).filterDate(start_date, end_date)
  	  poi_reduced_imgs1 = viirs1.map(poi_mean)
  	  nested_list1 = poi_reduced_imgs1.reduceColumns(ee.Reducer.toList(2), ['date',column_name]).values().get(0)
  	  # dont forget we need to call the callback method "getInfo" to retrieve the data
  	  df = pd.DataFrame(nested_list1.getInfo(), columns=['date',column_name])
      
  	  df['date'] = pd.to_datetime(df['date'])
  	  df = df.set_index('date')
  	  df['cell_id'] = current_cell_id
  	  df['latitude'] = latitude
  	  df['longitude'] = longitude
  	  #df.to_csv(single_csv_file)

  	  df_list = [all_cell_df, df]
  	  all_cell_df = pd.concat(df_list) # merge into big dataframe
      
    except Exception as e:
      
  	  print(e)
  	  pass
    
    
all_cell_df.to_csv(final_csv_file)  



