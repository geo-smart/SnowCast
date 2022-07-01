# Reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernel

# Standard Library
import os.path

# 3rd Party Packages
import ee
import pandas as pd

exit()  # comment to download new files

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()  # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Initialize()

# Data Information
org_name = 'sentinel2'
product_name = 'COPERNICUS/S2_SR'
start_date = '2017-03-28'
end_date = '2021-12-31'
variables = ['B11', 'B12', 'B2']
columns = ['swir_1', 'swir_2', 'blue']
var_name = None
column_name = None

# Directory Information
homedir = os.path.expanduser('~')
github_dir = f"{homedir}/Documents/GitHub/SnowCast"
station_cell_mapper_file = f"{github_dir}/data/ready_for_training/station_cell_mapping.csv"
dfolder = f"{homedir}/Documents/GitHub/SnowCast/data/sim_training/{org_name}/"

try:
    os.makedirs(dfolder)
except FileExistsError:
    pass


# Functions
def viirs_map(viirs, poi):
    def poi_mean(img):
        reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=30)
        mean = reducer.get(var_name)
        # print(img.date())
        # print(img.id().getInfo())
        date = img.date().format()
        return img.set('date', date).set(column_name, mean)

    def maskClouds(img):
        qa = img.select(var_name)
        cloudBitMask = 2 ** 10
        # cirrusBitMask = 2**11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0)  # .and(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return img.updateMask(mask).divide(10000).copyProperties(img, ['system:time_start'])

    return viirs.map(maskClouds).map(poi_mean)


def create_df(start_date, end_date, poi):
    viirs = ee.ImageCollection(product_name).filterDate(start_date, end_date).filterBounds(poi).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))  # .select(var_name)
    poi_reduced_imgs = viirs_map(viirs, poi)
    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date', column_name]).values().get(0)
    # dont forget we need to call the callback method "getInfo" to retrieve the data
    return pd.DataFrame(nested_list.getInfo(), columns=['date', column_name])


def main():
    global column_name, var_name

    # Read the Grid Geometry File
    station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)

    all_cells_df = {
        column_var: []  # list of pandas dataframes
        for column_var in columns
    }

    # Removes duplicate indices
    scmd = set(station_cell_mapper_df['cell_id'])

    for ind, current_cell_id in enumerate(scmd):

        # Location information
        longitude = station_cell_mapper_df['lon'][ind]
        latitude = station_cell_mapper_df['lat'][ind]

        print(f"{ind}/{len(scmd)}: collecting {current_cell_id}")  # logging

        # identify a 30 meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(longitude, latitude).buffer(30)

        for index in range(len(columns)):

            var_name = variables[index]
            column_name = columns[index]

            try:
                single_csv_file = f"{dfolder}/{column_name}_{current_cell_id}.csv"
                print(f"\ton column {column_name}")  # logging

                if os.path.exists(single_csv_file):
                    print("\t\texists skipping..")
                    continue

                df = create_df(start_date, end_date, poi)

                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                df['cell_id'] = current_cell_id
                df['latitude'] = latitude
                df['longitude'] = longitude
                df.to_csv(single_csv_file)

                all_cells_df[column_name].append(df)

                print("\t\tfinished")
            except Exception as e:
                print(f"\t\t{e}")

    print("\n\nsaving combined csv files")

    for column_name in all_cells_df:
        if len(all_cells_df[column_name]) > 0:
            pd.concat(all_cells_df[column_name]).to_csv(f"{dfolder}/{column_name}.csv")

    print("finished")


main()
