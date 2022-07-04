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

# Data information
org_name = 'modis'
product_name = f'MODIS/006/MOD10A1'
start_date1 = '2016-01-01'
end_date1 = '2017-12-31'
start_date2 = '2020-01-01'
end_date2 = '2021-12-31'
var_name = 'NDSI'
column_name = 'mod10a1_ndsi'

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
        return img.set('date', img.date().format()).set(column_name, mean)

    return viirs.map(poi_mean)


def create_df(start_date, end_date, poi):
    viirs = ee.ImageCollection(product_name).filterDate(start_date, end_date).filterBounds(poi).select(var_name)
    poi_reduced_imgs = viirs_map(viirs, poi)
    nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date', column_name]).values().get(0)
    # dont forget we need to call the callback method "getInfo" to retrieve the data
    return pd.DataFrame(nested_list.getInfo(), columns=['date', column_name])


def main():
    station_cell_mapper_df = pd.read_csv(station_cell_mapper_file)
    scmd = set(station_cell_mapper_df['cell_id'])

    all_cell_df = []
    for ind, current_cell_id in enumerate(scmd):
        try:
            print(f"{ind + 1}/{len(scmd)}: collecting {current_cell_id}")  # logging
            print(f"\ton column {column_name}")  # logging
            single_csv_file = f"{dfolder}/{column_name}_{current_cell_id}.csv"

            if os.path.exists(single_csv_file):
                print("\t\texists skipping..")
                continue

            longitude = station_cell_mapper_df['lon'][ind]
            latitude = station_cell_mapper_df['lat'][ind]

            # identify a 30 meter buffer around our Point Of Interest (POI)
            poi = ee.Geometry.Point(longitude, latitude).buffer(30)

            df1 = create_df(start_date1, end_date1, poi)
            df2 = create_df(start_date2, end_date2, poi)
            df = pd.concat([df1, df2])

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            df['cell_id'] = current_cell_id
            df['latitude'] = latitude
            df['longitude'] = longitude
            df.to_csv(single_csv_file)

            all_cell_df.append(df)  # merge into big dataframe

            print("\t\tfinished")

        except Exception as e:
            print(f"\t\t{e}")

    pd.concat(all_cell_df).to_csv(f"{dfolder}/{column_name}.csv")


main()
