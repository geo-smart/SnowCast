import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
from datetime import timedelta, datetime
import numpy as np
import pyproj
import uuid


reference_date = datetime(1900, 1, 1)
day_value = 44998
result_date = reference_date + timedelta(days=day_value)
current_datetime = result_date.strftime("%Y-%m-%d")
# current_datetime = datetime.now()
timestamp_string = current_datetime

def lat_lon_to_map_coordinates(lon, lat, m):
    x, y = m(lon, lat)
    return x, y

def convert_csvs_to_images():
    # Load the CSV data into a DataFrame
    data = pd.read_csv('/home/chetana/gridmet_test_run/test_data_prediected.csv')

    # Define the map boundaries for the Western US
    lon_min, lon_max = -125, -100
    lat_min, lat_max = 25, 49.5

    # Create the Basemap instance
    m = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max,
                projection='merc', resolution='i')

    # Convert lon/lat to map coordinates
    x, y = m(data['lon'].values, data['lat'].values)

    # Plot the data using vibrant colors based on predicted_swe
    plt.scatter(x, y, c=data['predicted_swe'], cmap='coolwarm', s=30, edgecolors='none', alpha=0.7)

    # Add colorbar for reference
    cbar = plt.colorbar()
    cbar.set_label('Predicted SWE')

    # Draw coastlines and other map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    reference_nc_file = nc.Dataset('/home/chetana/gridmet_test_run/gridmet_climatology/etr_2023.nc')

    reference_date = datetime(1900, 1, 1)
    day = reference_nc_file.variables['day'][:]
    day_value = day[-1]
    
    day_value = 44998
    
    result_date = reference_date + timedelta(days=day_value)
    today = result_date.strftime("%Y-%m-%d")
    

    # Add a title
    plt.title(f'Predicted SWE in the Western US - {today}', pad=20)

    # Add labels for latitude and longitude on x and y axes with smaller font size
    plt.xlabel('Longitude', fontsize=6)
    plt.ylabel('Latitude', fontsize=6)

    # Add longitude values to the x-axis and adjust font size
    x_ticks_labels = np.arange(lon_min, lon_max + 5, 5)
    x_tick_labels_str = [f"{lon:.1f}°W" if lon < 0 else f"{lon:.1f}°E" for lon in x_ticks_labels]
    plt.xticks(*m(x_ticks_labels, [lat_min] * len(x_ticks_labels)), fontsize=6)
    plt.gca().set_xticklabels(x_tick_labels_str)

    # Add latitude values to the y-axis and adjust font size
    y_ticks_labels = np.arange(lat_min, lat_max + 5, 5)
    y_tick_labels_str = [f"{lat:.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S" for lat in y_ticks_labels]
    plt.yticks(*m([lon_min] * len(y_ticks_labels), y_ticks_labels), fontsize=6)
    plt.gca().set_yticklabels(y_tick_labels_str)

    # Convert map coordinates to latitude and longitude for y-axis labels
    y_tick_positions = np.linspace(lat_min, lat_max, len(y_ticks_labels))
    y_tick_positions_map_x, y_tick_positions_map_y = lat_lon_to_map_coordinates([lon_min] * len(y_ticks_labels), y_tick_positions, m)
    y_tick_positions_lat, _ = m(y_tick_positions_map_x, y_tick_positions_map_y, inverse=True)
    y_tick_positions_lat_str = [f"{lat:.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S" for lat in y_tick_positions_lat]
    plt.yticks(y_tick_positions_map_y, y_tick_positions_lat_str, fontsize=6)

    plt.text(0.98, 0.02, 'Copyright © SWE Wormhole Team',
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gcf().transFigure, fontsize=6, color='black')

    # Adjust the bottom and top margins to create more white space between the title and the plot
    plt.subplots_adjust(bottom=0.15)
    # Show the plot or save it to a file
    plt.savefig(f'/home/chetana/gridmet_test_run/predicted_swe-{timestamp_string}-{uuid.uuid4().hex}.png')
    # plt.show()  # Uncomment this line if you want to display the plot directly instead of saving it to a file

convert_csvs_to_images()

