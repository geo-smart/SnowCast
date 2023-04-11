import pandas as pd
import xarray as xr
from loaders import ProgressLoader
from datetime import datetime, timedelta


start_date = datetime(2000, 1, 1)
end_date = datetime(2000, 1, 2)

filename = "output.csv"
header_written = False

all_cell_coords_path = "data/all_cell_coords_file.csv"
all_cell_coords_df = pd.read_csv(all_cell_coords_path)

submission_eval_path = "data/submission_format_eval.csv"
submission_eval_df = pd.read_csv(submission_eval_path)

df = pd.DataFrame(columns=['date', 'tmmx', 'tmmn', 'pr', 'vpd', 'eto', 'rmax', 'rmin', 'vs', 'cell_id', 'latitude',
                           'longitude'])

loader = ProgressLoader(total=len(submission_eval_df) * (int((end_date - start_date).days) + 1))

for idx, row in submission_eval_df.iterrows():
    longitude = all_cell_coords_df['lon'][idx]
    latitude = all_cell_coords_df['lat'][idx]

    for n in range((end_date - start_date).days + 1):
        current_date = start_date + timedelta(n)
        data_files = {
            'tmmx': xr.open_dataset(f'data/tmmx_{current_date.year}.nc'),
            'tmmn': xr.open_dataset(f'data/tmmn_{current_date.year}.nc'),
            'pr': xr.open_dataset(f'data/pr_{current_date.year}.nc'),
            'vpd': xr.open_dataset(f'data/vpd_{current_date.year}.nc'),
            'pet': xr.open_dataset(f'data/pet_{current_date.year}.nc'),
            'rmax': xr.open_dataset(f'data/rmax_{current_date.year}.nc'),
            'rmin': xr.open_dataset(f'data/rmin_{current_date.year}.nc'),
            'vs': xr.open_dataset(f'data/vs_{current_date.year}.nc'),
        }
        row_builder = {
            'date': current_date,
            'cell_id': all_cell_coords_df['cell_id'][idx],
            'latitude': latitude,
            'longitude': longitude,
        }

        for variable, data in data_files.items():
            selector = str()

            if variable == "tmmx" or variable == "tmmn":
                selector = "air_temperature"
            if variable == "pr":
                selector = "precipitation_amount"
            if variable == "vpd":
                selector = "mean_vapor_pressure_deficit"
            if variable == "pet":
                selector = "potential_evapotranspiration"
            if variable == "rmax" or variable == "rmin":
                selector = "relative_humidity"
            if variable == "vs":
                selector = "wind_speed"

            tmp = data.sel(lat=latitude, lon=longitude, day=current_date, method='nearest')
            val = tmp[selector].values
            row_builder[variable] = val
            # values = data.sel(lat=latitude, lon=longitude, day=current_date, method='nearest')[selector].values
            # row_builder[variable] = values.item() if np.isscalar(values) else values[0]
        df = df.append(row_builder, ignore_index=True)
        if not header_written:
            with open(filename, 'w') as f:
                f.write('date,cell_id,latitude,longitude,tmmx,tmmn,pr,vpd,pet,rmax,rmin,vs\n')
                header_written = True
        print(row_builder)
        # write row to file
        with open(filename, 'a') as f:
            f.write(
                f"{row_builder['date']},{row_builder['cell_id']},{row_builder['latitude']},{row_builder['longitude']},{row_builder['tmmx']},{row_builder['tmmn']},{row_builder['pr']},{row_builder['vpd']},{row_builder['pet']},{row_builder['rmax']},{row_builder['rmin']},{row_builder['vs']}\n")

        loader.progress()
# df.to_csv('test.csv', index=False)


