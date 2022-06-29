# Reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernel

# Standard Library
import os.path

# 3rd Party Packages
import ee
import pandas as pd

# exit()  # comment to download new files

try:
    ee.Initialize()
except Exception:
    ee.Authenticate()  # this must be run in terminal instead of Geoweaver. Geoweaver doesn't support prompt.
    ee.Initialize()

# Data information
org_name = 'gridmet'
product_name = 'IDAHO_EPSCOR/GRIDMET'
start_date = '2021-01-01'
end_date = '2021-12-31'
column_name = None
var_name = None
columns = ['tmmn', 'tmmx', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs']

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
        reducer = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=1000)
        mean = reducer.get(var_name)
        return img.set('date', img.date().format()).set(column_name, mean)

    return viirs.map(poi_mean)


def create_df(start_date, end_date, poi):
    viirs = ee.ImageCollection(product_name).filterDate(start_date, end_date)
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
    scmd = set(station_cell_mapper_df.index)

    for ind in scmd:

        # Location information
        current_cell_id = station_cell_mapper_df['cell_id'][ind]
        longitude = station_cell_mapper_df['lon'][ind]
        latitude = station_cell_mapper_df['lat'][ind]

        print(f"{ind}/{len(scmd)}: collecting {current_cell_id}")  # logging

        # Identify a 500 meter buffer around our Point Of Interest (POI)
        poi = ee.Geometry.Point(longitude, latitude).buffer(1000)

        for column_name in columns:
            var_name = column_name

            try:
                single_csv_file = f"{dfolder}/{column_name}_{current_cell_id}.csv"
                print(f"\ton column {column_name}")  # logging

                if os.path.exists(single_csv_file):
                    print("\t\texists skipping..")  # logging
                    continue

                df = create_df(start_date, end_date, poi)

                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                df['cell_id'] = current_cell_id
                df['latitude'] = latitude
                df['longitude'] = longitude
                df.to_csv(single_csv_file)

                all_cells_df[column_name].append(df)

                print("\t\tfinished")  # logging
            except Exception as e:
                print(f"\t\t{e}")  # logging

    print("\n\nsaving combined csv files")

    for column_name, dfs in all_cells_df:
        pd.concat(dfs).to_csv(f"{dfolder}/{column_name}.csv")

    print("finished")


main()
