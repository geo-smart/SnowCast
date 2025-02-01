import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import numpy as np
import uuid
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio import warp
from shapely.geometry import Point
from rasterio.crs import CRS
import rasterio.features
from rasterio.features import rasterize
import os
import math
from datetime import datetime, timedelta

from scipy.interpolate import griddata

# Import utility functions and variables from 'snowcast_utils'
from snowcast_utils import data_dir, work_dir, plot_dir, output_dir, test_start_date, test_end_date,process_dates_in_range

# Define a custom colormap with specified colors and ranges
colors = [
    (0.8627, 0.8627, 0.8627),  # #DCDCDC - 0 - 1
    (0.8627, 1.0000, 1.0000),  # #DCFFFF - 1 - 2
    (0.6000, 1.0000, 1.0000),  # #99FFFF - 2 - 4
    (0.5569, 0.8235, 1.0000),  # #8ED2FF - 4 - 6
    (0.4509, 0.6196, 0.8745),  # #739EDF - 6 - 8
    (0.4157, 0.4706, 1.0000),  # #6A78FF - 8 - 10
    (0.4235, 0.2784, 1.0000),  # #6C47FF - 10 - 12
    (0.5529, 0.0980, 1.0000),  # #8D19FF - 12 - 14
    (0.7333, 0.0000, 0.9176),  # #BB00EA - 14 - 16
    (0.8392, 0.0000, 0.7490),  # #D600BF - 16 - 18
    (0.7569, 0.0039, 0.4549),  # #C10074 - 18 - 20
    (0.6784, 0.0000, 0.1961),  # #AD0032 - 20 - 30
    (0.5020, 0.0000, 0.0000)   # #800000 - > 30
]

cmap_name = 'custom_snow_colormap'
custom_cmap = mcolors.ListedColormap(colors)

lon_min, lon_max = -125, -100
lat_min, lat_max = 25, 49.5

# Define value ranges for color mapping
fixed_value_ranges = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30]

import os
import requests
import zipfile

def download_and_unzip_shapefile(url, output_dir):
    """
    Download and unzip a shapefile from the given URL.

    Args:
        url (str): URL of the zip file containing the shapefile.
        output_dir (str): Directory to save the downloaded file and its contents.

    Returns:
        str: Path to the extracted shapefile directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the file paths
    zip_file_path = os.path.join(output_dir, "tl_2023_us_state.zip")

    print(f"Step 1: Downloading shapefile from {url}...")
    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error if the download fails
    with open(zip_file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Step 2: Download complete. Saved to {zip_file_path}")

    # Extract the zip file
    print(f"Step 3: Extracting files to {output_dir}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Step 4: Extraction complete.")
    return output_dir

def retrieve_state_boundary():
    """
    Retrieve the state boundary shapefile.
    Downloads and extracts the file only if it doesn't already exist.

    Returns:
        str: Absolute path to the main shapefile.
    """
    # URL of the shapefile zip
    shapefile_url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"

    # Output directory for the downloaded and extracted files
    output_directory = os.path.join(data_dir, "shapefiles", "tl_2023_us_state")
    os.makedirs(output_directory, exist_ok=True)

    # Path to the main shapefile
    shapefile_path = os.path.join(output_directory, "tl_2023_us_state.shp")

    # Check if the shapefile already exists
    if os.path.exists(shapefile_path):
        print(f"Shapefile already exists at: {shapefile_path}. Skipping download.")
    else:
        print("Shapefile not found. Downloading...")
        # Download and unzip the shapefile
        extracted_dir = download_and_unzip_shapefile(shapefile_url, output_directory)
        print(f"Shapefile extracted to: {extracted_dir}")
    
    return os.path.abspath(shapefile_path)

# Define the lat_lon_to_map_coordinates function
def lat_lon_to_map_coordinates(lon, lat, m):
    """
    Convert latitude and longitude coordinates to map coordinates.

    Args:
        lon (float or array-like): Longitude coordinate(s).
        lat (float or array-like): Latitude coordinate(s).
        m (Basemap): Basemap object representing the map projection.

    Returns:
        tuple: Tuple containing the converted map coordinates (x, y).
    """
    x, y = m(lon, lat)
    return x, y



def create_color_maps_with_value_range(df_col, value_ranges=None):
    """
    Create a colormap for value ranges and map data values to colors.

    Args:
        df_col (pd.Series): A Pandas Series containing data values.
        value_ranges (list, optional): A list of value ranges for color mapping.
            If not provided, the ranges will be determined automatically.

    Returns:
        tuple: Tuple containing the color mapping and the updated value ranges.
    """
    new_value_ranges = value_ranges
    if value_ranges is None:
        max_value = df_col.max()
        min_value = df_col.min()
        if min_value < 0:
            min_value = 0
        step_size = (max_value - min_value) / 12

        # Create 10 periods
        new_value_ranges = [min_value + i * step_size for i in range(12)]
    
    #print("new_value_ranges: ", new_value_ranges)
  
    # Define a custom function to map data values to colors
    def map_value_to_color(value):
        # Iterate through the value ranges to find the appropriate color index
        for i, range_max in enumerate(new_value_ranges):
            if value <= range_max:
                return colors[i]

        # If the value is greater than the largest range, return the last color
        return colors[-1]

    # Map predicted_swe values to colors using the custom function
    color_mapping = [map_value_to_color(value) for value in df_col.values]
    return color_mapping, new_value_ranges

def convert_csvs_to_images(input_csv: str = None, new_plot_path: str = None):
    """
    Convert CSV data to images with color-coded SWE predictions.

    Returns:
        None
    """
    global fixed_value_ranges
    if input_csv is None:
        input_csv = f"{work_dir}/test_data_predicted_n97KJ.csv"
    
    data = pd.read_csv(input_csv)
    print("statistic of predicted_swe: ", data['predicted_swe'].describe())
    data['predicted_swe'].fillna(0, inplace=True)
    
    for column in data.columns:
        column_data = data[column]
        print(column_data.describe())
    
    # Create a figure with a white background
    fig = plt.figure(facecolor='white')

    

    m = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                projection='merc', resolution='i')

    if "Latitude" in data.columns and "Longitude" in data.columns:
        data.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)

    x, y = m(data['lon'].values, data['lat'].values)
    print(data.columns)

    color_mapping, value_ranges = create_color_maps_with_value_range(data["predicted_swe"], fixed_value_ranges)
    
    # Plot the data using the custom colormap
    plt.scatter(x, y, c=color_mapping, cmap=custom_cmap, s=30, edgecolors='none', alpha=0.7)
    
    # Draw coastlines and other map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    reference_date = datetime(1900, 1, 1)
    day_value = day_index
    
    result_date = reference_date + timedelta(days=day_value)
    today = result_date.strftime("%Y-%m-%d")
    timestamp_string = result_date.strftime("%Y-%m-%d")
    
    # Add a title
    plt.title(f'Predicted SWE in the Western US - {today}', pad=20)

    # Add labels for latitude and longitude on x and y axes with smaller font size
    plt.xlabel('Longitude', fontsize=6)
    plt.ylabel('Latitude', fontsize=6)

    # Add longitude values to the x-axis and adjust font size
    x_ticks_labels = np.arange(lon_min, lon_max + 5, 5)
    x_tick_labels_str = [f"{lon:.1f}??W" if lon < 0 else f"{lon:.1f}??E" for lon in x_ticks_labels]
    plt.xticks(*m(x_ticks_labels, [lat_min] * len(x_ticks_labels)), fontsize=6)
    plt.gca().set_xticklabels(x_tick_labels_str)

    # Add latitude values to the y-axis and adjust font size
    y_ticks_labels = np.arange(lat_min, lat_max + 5, 5)
    y_tick_labels_str = [f"{lat:.1f}??N" if lat >= 0 else f"{abs(lat):.1f}??S" for lat in y_ticks_labels]
    plt.yticks(*m([lon_min] * len(y_ticks_labels), y_ticks_labels), fontsize=6)
    plt.gca().set_yticklabels(y_tick_labels_str)

    # Convert map coordinates to latitude and longitude for y-axis labels
    y_tick_positions = np.linspace(lat_min, lat_max, len(y_ticks_labels))
    y_tick_positions_map_x, y_tick_positions_map_y = lat_lon_to_map_coordinates([lon_min] * len(y_ticks_labels), y_tick_positions, m)
    y_tick_positions_lat, _ = m(y_tick_positions_map_x, y_tick_positions_map_y, inverse=True)
    y_tick_positions_lat_str = [f"{lat:.1f}??N" if lat >= 0 else f"{abs(lat):.1f}??S" for lat in y_tick_positions_lat]
    plt.yticks(y_tick_positions_map_y, y_tick_positions_lat_str, fontsize=6)

    # Create custom legend elements using the same colormap
    legend_elements = [Patch(color=colors[i], label=f"{value_ranges[i]} - {value_ranges[i+1]-1}" if i < len(value_ranges) - 1 else f"> {value_ranges[-1]}") for i in range(len(value_ranges))]

    # Create the legend outside the map
    legend = plt.legend(handles=legend_elements, loc='upper left', title='Legend', fontsize=8)
    legend.set_bbox_to_anchor((1.01, 1)) 

    # Remove the color bar
    #plt.colorbar().remove()

    plt.text(0.98, 0.02, 'Copyright ?? SWE Wormhole Team',
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gcf().transFigure, fontsize=6, color='black')

    # Set the aspect ratio to 'equal' to keep the plot at the center
    plt.gca().set_aspect('equal', adjustable='box')

    # Adjust the bottom and top margins to create more white space between the title and the plot
    plt.subplots_adjust(bottom=0.15, right=0.80)  # Adjust right margin to accommodate the legend
    # Show the plot or save it to a file
    if new_plot_path is None:
        new_plot_path = f'{work_dir}/predicted_swe-{test_start_date}.png'
    
    print(f"The new plot is saved to {new_plot_path}")
    plt.savefig(new_plot_path)
    # plt.show()  # Uncomment this line if you want to display the plot directly instead of saving it to a file

def plot_all_variables_in_one_csv(csv_path, res_png_path, target_date = test_start_date):
    result_var_df = pd.read_csv(csv_path)
    # Convert the 'date' column to datetime
    result_var_df['date'] = pd.to_datetime(result_var_df['date'])
    result_var_df.rename(
      columns={
        'Latitude': 'lat', 
        'Longitude': 'lon',
        'gridmet_lat': 'lat',
        'gridmet_lon': 'lon',
      }, 
      inplace=True)
    
  	# Create subplots with a number of rows based on the number of columns in the DataFrame
    
    us_boundary = gpd.read_file(retrieve_state_boundary())
    us_boundary_clipped = us_boundary.cx[lon_min:lon_max, lat_min:lat_max]
	
    lat_col = result_var_df[["lat"]]
    lon_col = result_var_df[["lon"]]
    print("lat_col.values = ", lat_col["lat"].values)
#     if "lat" == column_name or "lon" == column_name or "date" == column_name:
    columns_to_remove = [ "date", "Latitude", "Longitude", "gridmet_lat", "gridmet_lon", "lat", "lon"]

    # Check if each column exists before removing it
    for col in columns_to_remove:
        if col in result_var_df.columns:
            result_var_df = result_var_df.drop(columns=col)
        else:
            print(f"Column '{col}' not found in DataFrame.")
    
    print("result_var_df.shape: ", result_var_df.shape)
    print("result_var_df.head: ", result_var_df.head())
    
    
    num_columns = len(result_var_df.columns)  # don't plot lat and lon
    fig_width = 7 * num_columns  # You can adjust this multiplier based on your preference
    num_variables = len(result_var_df.columns)
    num_cols = int(math.sqrt(num_variables))
    num_rows = math.ceil(num_variables / num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*7, num_rows*6))
    
    
    # Flatten the axes array to simplify indexing
    axes = axes.flatten()
    
  	# Plot each variable in a separate subplot
    for i, column_name in enumerate(result_var_df.columns):
  	    print(f"Plot {column_name}")
  	    if column_name in ["lat", "lon"]:
  	        continue
        
        # Filter the DataFrame based on the target date
  	    result_var_df[column_name] = pd.to_numeric(result_var_df[column_name], errors='coerce')
  	    
  	    colormaplist, value_ranges = create_color_maps_with_value_range(result_var_df[column_name], fixed_value_ranges)
  	    scatter_plot = axes[i].scatter(
            lon_col["lon"].values, 
  	        lat_col["lat"].values, 
            label=column_name, 
            c=result_var_df[column_name], 
            cmap='viridis', 
              #s=200, 
            s=10, 
            marker='s',
            edgecolor='none',
        )
        
        # Add a colorbar
  	    cbar = plt.colorbar(scatter_plot, ax=axes[i])
  	    cbar.set_label(column_name)  # Label for the colorbar
        
        # Add boundary over the figure
  	    us_boundary_clipped.plot(ax=axes[i], color='none', edgecolor='black', linewidth=1)

        # Add labels and a legend
  	    axes[i].set_xlabel('Longitude')
  	    axes[i].set_ylabel('Latitude')
  	    axes[i].set_title(column_name+" - "+target_date)  # You can include target_date if needed
  	    axes[i].legend(loc='lower left')
    
    # Remove any empty subplots
    for i in range(num_variables, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(res_png_path)
    print(f"test image is saved at {res_png_path}")
    plt.close()
    
    
def plot_all_variables_in_one_figure_for_date(target_date=test_start_date):
  	selected_date = datetime.strptime(target_date, "%Y-%m-%d")
  	test_csv = f"{output_dir}/test_data_predicted_latest.csv"
  	res_png_path = f"{plot_dir}/{str(selected_date.year)}_all_variables_{target_date}.png"
  	plot_all_variables_in_one_csv(test_csv, res_png_path, target_date)
    
def convert_csvs_to_images_simple(
    target_date=test_start_date, 
    column_name = "predicted_swe", 
    test_csv: str = None,
    res_png_path: str = None,
):
    """
    Convert CSV data to simple scatter plot images for predicted SWE.

    Returns:
        None
    """
    
    selected_date = datetime.strptime(target_date, "%Y-%m-%d")
    var_name = column_name
    if test_csv is None:
        test_csv = f"{output_dir}/test_data_predicted_latest_{target_date}.csv_snodas_mask.csv"

    # Extract the directory from the target path
    target_plot_dir = os.path.dirname(test_csv)

    # Create all layers of directories if they don't exist
    os.makedirs(target_plot_dir, exist_ok=True)

    if res_png_path is None:
        res_png_path = f"{plot_dir}/{str(selected_date.year)}_{var_name}_{target_date}.png"
    
    result_var_df = pd.read_csv(test_csv)
    # Convert the 'date' column to datetime
    if 'date_x' in result_var_df.columns and 'date_y' in result_var_df.columns:
        # Drop one of the date columns (let's drop 'date_y')
        result_var_df.drop(columns=['date_y'], inplace=True)
        
        # Rename 'date_x' to 'date'
        result_var_df.rename(columns={'date_x': 'date'}, inplace=True)
    
    if 'date' in result_var_df.columns:
        result_var_df['date'] = pd.to_datetime(result_var_df['date'])

    # Filter the DataFrame based on the target date
    result_var_df[var_name] = pd.to_numeric(result_var_df[var_name], errors='coerce')
    
    colormaplist, value_ranges = create_color_maps_with_value_range(result_var_df[var_name], fixed_value_ranges)

    if "Latitude" in result_var_df.columns and "Longitude" in result_var_df.columns:
        result_var_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)

    # Create a scatter plot
    plt.scatter(result_var_df["lon"].values, 
                result_var_df["lat"].values, 
                label=column_name, 
                c=result_var_df[column_name], 
                cmap='viridis', 
                #s=200, 
                s=10, 
                marker='s',
                edgecolor='none',
               )

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label(column_name)  # Label for the colorbar
    
    # Add labels and a legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'{column_name} - {target_date}')
    plt.legend(loc='lower left')
    
    us_boundary = gpd.read_file(retrieve_state_boundary())
    us_boundary_clipped = us_boundary.cx[lon_min:lon_max, lat_min:lat_max]

    us_boundary_clipped.plot(ax=plt.gca(), color='none', edgecolor='black', linewidth=1)

    
    plt.savefig(res_png_path)
    print(f"test image is saved at {res_png_path}")
    plt.close()

def convert_csv_to_geotiff(
    target_date,
    test_csv: str = None,
    target_geotiff_file: str = None,
    output_dir: str = ".",
    resolution: float = 0.01,  # Define the grid resolution
):
    # Load your CSV file
    if test_csv is None:
        test_csv = f"{output_dir}/test_data_predicted_latest_{target_date}.csv"
    
    result_var_df = pd.read_csv(test_csv)
    result_var_df.rename(
        columns={
            'Latitude': 'lat',
            'Longitude': 'lon',
            'gridmet_lat': 'lat',
            'gridmet_lon': 'lon',
        },
        inplace=True
    )

    # Specify the output GeoTIFF file
    if target_geotiff_file is None:
        target_geotiff_file = f"{output_dir}/swe_predicted_{target_date}.tif"

    target_plot_dir = os.path.dirname(target_geotiff_file)
    os.makedirs(target_plot_dir, exist_ok=True)

    # Extract latitude, longitude, and variable of interest
    df = result_var_df[["lat", "lon", "predicted_swe"]]
    latitude = df['lat'].values
    longitude = df['lon'].values
    swe = df['predicted_swe'].values

    # Define raster grid bounds and resolution
    lat_min, lat_max = latitude.min(), latitude.max()
    lon_min, lon_max = longitude.min(), longitude.max()

    # Create the raster grid
    lon_grid, lat_grid = np.meshgrid(
        np.arange(lon_min, lon_max, resolution),
        np.arange(lat_min, lat_max, resolution)
    )

    # Interpolate data to the grid
    grid_swe = griddata(
        (longitude, latitude),  # Input coordinates
        swe,                   # Input values
        (lon_grid, lat_grid),  # Grid coordinates
        method='linear'        # Interpolation method: 'linear', 'nearest', 'cubic'
    )

    # Flip the grid vertically to align with raster conventions
    grid_swe = np.flipud(grid_swe)

    # Define transform for the raster
    transform = from_origin(
        lon_min, lat_max,  # Upper-left corner
        resolution, resolution  # Pixel size
    )

    # Save the GeoTIFF
    with rasterio.open(
        target_geotiff_file,
        'w',
        driver='GTiff',
        height=grid_swe.shape[0],
        width=grid_swe.shape[1],
        count=1,  # Number of bands
        dtype=grid_swe.dtype,
        crs="EPSG:4326",  # WGS84 Latitude/Longitude
        transform=transform
    ) as dst:
        dst.write(grid_swe, 1)  # Write the raster data to the first band

    print(f"GeoTIFF saved to {target_geotiff_file}")

def process_swe_prediction(current_date):
    """
    Example callback function to process SWE prediction for a specific date.

    Args:
        current_date (datetime): The date to process.
        output_dir (str): Directory for output files.
        plot_dir (str): Directory for plot files.
    """
    current_date_str = current_date.strftime("%Y-%m-%d")
    test_csv = f"{output_dir}/test_data_predicted_latest_{current_date_str}.csv_snodas_mask.csv"

    if not os.path.exists(test_csv):
        print(f"Warning: {test_csv} is missing. Skipping this day.")
        return

    # Example processing steps
    convert_csvs_to_images_simple(current_date_str, test_csv=test_csv)
    convert_csv_to_geotiff(
        current_date_str,
        test_csv=test_csv,
        target_geotiff_file=f"{output_dir}/swe_predicted_{current_date_str}.tif",
    )


if __name__ == "__main__":
    # Uncomment the function call you want to use:
    #convert_csvs_to_images()
    
    #test_start_date = "2022-10-09"
    process_dates_in_range(
        # start_date="2025-01-14",
        # end_date="2025-01-14",
        start_date=test_start_date,
        end_date=test_end_date,
        days_look_back=0,
        callback=process_swe_prediction,
    )


