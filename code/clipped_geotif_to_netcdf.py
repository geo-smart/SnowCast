import xarray as xr
import rioxarray as rxr
import os
import re
import pandas as pd
import glob

def convert_geotiffs_to_netcdf(folder_path, output_filename="swe_predicted.nc"):
    """
    Convert all GeoTIFF files in the specified folder matching the pattern
    'clipped_swe_predicted_YYYY-MM-DD.tif' into a NetCDF file with dimensions (time, lat, lon).
    
    Parameters:
    - folder_path (str): Path to the folder containing GeoTIFF files.
    - output_filename (str): Name of the output NetCDF file (default: "swe_predicted.nc").
    """

    # Find all matching GeoTIFF files
    tif_files = sorted(glob.glob(os.path.join(folder_path, "clipped_swe_predicted_*.tif")))
    
    if not tif_files:
        print("No matching GeoTIFF files found.")
        return

    # Extract dates from filenames
    def extract_date(filename):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        return match.group(1) if match else None

    dates = [extract_date(os.path.basename(f)) for f in tif_files]
    dates = pd.to_datetime(dates)

    # Load and stack all GeoTIFFs
    data_arrays = []
    for file, date in zip(tif_files, dates):
        # print("date = ", date, file)
        da = rxr.open_rasterio(file, masked=True).squeeze()  # Load and remove single band dimension
        da = da.assign_coords(time=date)  # Add time coordinate
        da.name = "SWE"  # Set variable name
        data_arrays.append(da)

    # Merge into a single dataset
    ds = xr.concat(data_arrays, dim="time")

    # Save to NetCDF
    output_path = os.path.join(folder_path, output_filename)
    ds.to_netcdf(output_path)

    print(f"NetCDF file saved: {output_path}")

def main():
    convert_geotiffs_to_netcdf(
        folder_path = "/home/chetana/data/output/TuolumneBasin_clipped_shp",
        output_filename = "TuolumneBasin_swe_predicted.nc"
    )
    
    convert_geotiffs_to_netcdf(
        folder_path = "/home/chetana/data/output/skagitboundary_clipped_geojson",  # Change this to your actual folder path
        output_filename = "skagitboundary_swe_predicted.nc"
    )

if __name__ == "__main__":
    main()
