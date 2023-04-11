from all_dependencies import *
from datetime import date
from snowcast_utils import *
import traceback
import dask.dataframe as dd
from dask import delayed, compute

try:
    ee.Initialize()
except Exception as e:
    # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Authenticate()
    ee.Initialize()

# read the grid geometry file
homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"

# read grid cell
submission_format_file = f"{github_dir}/data/snowcast_provided/submission_format_eval.csv"
submission_format_df = pd.read_csv(
    submission_format_file, header=0, index_col=0)

all_cell_coords_file = f"{github_dir}/data/snowcast_provided/all_cell_coords_file.csv"
all_cell_coords_df = pd.read_csv(all_cell_coords_file, header=0, index_col=0)

org_name = 'modis'
product_name = f'MODIS/006/MOD10A1'
var_name = 'NDSI'
column_name = 'mod10a1_ndsi'

# start_date = "2022-04-20"#test_start_date
start_date = findLastStopDate(
    f"{github_dir}/data/sat_testing/modis", "%Y-%m-%d")
end_date = test_end_date

final_csv_file = f"{homedir}/Documents/GitHub/SnowCast/data/sat_testing/{org_name}/{column_name}_{start_date}_{end_date}.csv"
print(f"Results will be saved to {final_csv_file}")

if os.path.exists(final_csv_file):
    os.remove(final_csv_file)

# Set up batch processing parameters
batch_size = 50  # Number of cells to query in each batch
max_retries = 5  # Maximum number of times to retry a batch that fails
wait_time = 30  # Number of seconds to wait before retrying a failed batch


def process_batch(cell_ids):
    # Process a batch of cells

    all_cell_df = pd.DataFrame(columns=['date', column_name, 'cell_id'])

    for current_cell_id in cell_ids:

        try:
            longitude = all_cell_coords_df['lon'][current_cell_id]
            latitude = all_cell_coords_df['lat'][current_cell_id]

            # identify a 20 meter buffer around our Point Of Interest (POI)
            poi = ee.Geometry.Point(longitude, latitude).buffer(20)

            def poi_mean(img):
                reducer = img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=poi, scale=30)
                mean = reducer.get(var_name)
                return img.set('date', img.date().format()).set(column_name, mean)

            viirs1 = ee.ImageCollection(
                product_name).filterDate(start_date, end_date)
            poi_reduced_imgs1 = viirs1.map(poi_mean)
            nested_list1 = poi_reduced_imgs1.reduceColumns(
                ee.Reducer.toList(2), ['date', column_name]).values().get(0)

            # dont forget we need to call the callback method "getInfo" to retrieve the data
            df = pd.DataFrame(nested_list1.getInfo(),
                              columns=['date', column_name])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['cell_id'] = current_cell_id

            df_list = [all_cell_df, df]
            all_cell_df = pd.concat(df_list)  # merge into big dataframe

        except Exception as e:
            print(f"Error processing cell {current_cell_id}: {e}")
            traceback.print_exc()

            # Retry failed batch up to max_retries times
            retries = 0
            while retries < max_retries:
                print(
                    f"Retrying batch for cell {current_cell_id} in {wait_time} seconds... (retry {retries+1}/{max_retries})")
                time.sleep(wait_time)
                try:
                    df = process_batch([current_cell_id])
                    df_list = [all_cell_df, df]
                    # merge into big dataframe
                    all_cell_df = pd.concat(df_list)
                    print(
                        f"Batch for cell {current_cell_id} processed successfully after {retries+1} retries.")
                    break
                except Exception as e:
                    print(f"Error processing cell {current_cell_id}: {e}")
                    traceback.print_exc()
                    retries += 1
            else:
                print(
                    f"Batch for cell {current_cell_id} failed after {max_retries} retries.")

    return all_cell_df


# Split cell IDs into batches
cell_ids = submission_format_df.index.tolist()
batches = [cell_ids[i:i+batch_size]
           for i in range(0, len(cell_ids), batch_size)]

# Process batches using Dask
delayed_results = [delayed(process_batch)(batch) for batch in batches]
all_results = compute(*delayed_results, scheduler='processes')

# Concatenate all results
df_list = []
for result in all_results:
    df_list.append(result)

final_df = pd.concat(df_list)

# Save results to CSV file
final_df.to_csv(final_csv_file, header=True, index=True)

print(
    f"Batch processing completed successfully. Results saved to {final_csv_file}.")

