# This script will download modis data for all the testing sites from Google Earth Engine.
# The start date is the last stop date of the last run.

import traceback

from snowcast_utils import *

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()  # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Initialize()

# read the grid geometry file
homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
# read grid cell
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"
submission_format_df = pd.read_csv(submission_format_file, header=0, index_col=0)

all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
all_cell_coords_df = pd.read_csv(all_cell_coords_file, header=0, index_col=0)

org_name = 'modis'
product_name = f'MODIS/006/MOD10A1'
var_name = 'NDSI'
column_name = 'mod10a1_ndsi'
# start_date = "2022-04-20"#test_start_date
start_date = findLastStopDate(f"{github_dir}/data/sat_testing/modis", "%Y-%m-%d")
end_date = test_end_date

final_csv_file = f"{homedir}/Documents/GitHub/SnowCast/data/sat_testing/{org_name}/{column_name}_{start_date}_{end_date}.csv"
print(f"Results will be saved to {final_csv_file}")

if os.path.exists(final_csv_file):
    # print("exists exiting..")
    # exit()
    os.remove(final_csv_file)

all_cell_df = pd.DataFrame(columns=['date', column_name, 'cell_id', 'latitude', 'longitude'])
print("start to traverse the cells in submission_format_eval.csv..")

for current_cell_id in submission_format_df.index:

    try:

        longitude = all_cell_coords_df['lon'][current_cell_id]
        latitude = all_cell_coords_df['lat'][current_cell_id]

        # identify a 500 meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(longitude, latitude).buffer(30)


        def poi_mean(img):
            reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30)
            mean = reducer.get(var_name)
            return img.set('date', img.date().format()).set(column_name, mean)


        viirs1 = ee.ImageCollection(product_name).filterDate(start_date, end_date)
        poi_reduced_imgs1 = viirs1.map(poi_mean)
        nested_list1 = poi_reduced_imgs1.reduceColumns(ee.Reducer.toList(2), ['date', column_name]).values().get(0)
        # dont forget we need to call the callback method "getInfo" to retrieve the data
        df = pd.DataFrame(nested_list1.getInfo(), columns=['date', column_name])

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['cell_id'] = current_cell_id
        df['latitude'] = latitude
        df['longitude'] = longitude
        # df.to_csv(single_csv_file)

        df_list = [all_cell_df, df]
        all_cell_df = pd.concat(df_list)  # merge into big dataframe

    except Exception as e:
        print(traceback.format_exc())
        print("failed", e)
        pass

all_cell_df.to_csv(final_csv_file)

print(f"All points have been saved to {final_csv_file}")
