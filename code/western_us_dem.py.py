import numpy as np
import pandas as pd
from osgeo import gdal
import warnings
import rasterio
import csv
from rasterio.transform import Affine
from scipy.ndimage import sobel, gaussian_filter

# Set the warning filter globally to ignore the FutureWarning
warnings.simplefilter("ignore", FutureWarning)

def lat_lon_to_pixel(lat, lon, geotransform):
    x = int((lon - geotransform[0]) / geotransform[1])
    y = int((lat - geotransform[3]) / geotransform[5])
    return x, y


def calculate_slope_aspect_for_single(elevation_data, pixel_size_x, pixel_size_y):
    # Calculate slope using the Sobel operator
    slope_x = np.gradient(elevation_data, pixel_size_x, axis=1)
    slope_y = np.gradient(elevation_data, pixel_size_y, axis=0)
    slope_rad = np.arctan(np.sqrt(slope_x ** 2 + slope_y ** 2))
    slope_deg = np.degrees(slope_rad)

    # Calculate aspect (direction of the steepest descent)
    aspect_rad = np.arctan2(slope_y, -slope_x)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

    return slope_deg, aspect_deg


def save_as_geotiff(data, output_file, src_file):
    with rasterio.open(src_file) as src_dataset:
        profile = src_dataset.profile
        transform = src_dataset.transform

        # Update the data type, count, and set the transform for the new dataset
        profile.update(dtype=rasterio.float32, count=1, transform=transform)

        # Create the new GeoTIFF file
        with rasterio.open(output_file, 'w', **profile) as dst_dataset:
            # Write the data to the new GeoTIFF
            dst_dataset.write(data, 1)
  

def calculate_slope_aspect(dem_file):
    with rasterio.open(dem_file) as dataset:
        # Read the DEM data as a numpy array
        dem_data = dataset.read(1)

        # Get the geotransform to convert pixel coordinates to geographic coordinates
        transform = dataset.transform

        # Calculate the slope and aspect using numpy
        dx, dy = np.gradient(dem_data, transform[0], transform[4])
        slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2)) * (180.0 / np.pi)
        aspect = np.arctan2(-dy, dx) * (180.0 / np.pi)

        # Adjust aspect values to range from 0 to 360 degrees
        aspect[aspect < 0] += 360
        print(f"slope shape: {slope.shape}")
        print(f"aspect shape: {aspect.shape}")
        
        
    return slope, aspect
  
def calculate_curvature(elevation_data, pixel_size_x, pixel_size_y):
    # Calculate curvature using the Laplacian operator
    curvature_x = np.gradient(np.gradient(elevation_data, pixel_size_x, axis=1), pixel_size_x, axis=1)
    curvature_y = np.gradient(np.gradient(elevation_data, pixel_size_y, axis=0), pixel_size_y, axis=0)
    curvature = curvature_x + curvature_y

    return curvature

  
def calculate_curvature(dem_file, sigma=1):
    with rasterio.open(dem_file) as dataset:
        # Read the DEM data as a numpy array
        dem_data = dataset.read(1)

        # Calculate the gradient using the Sobel filter
        dx = sobel(dem_data, axis=1, mode='constant')
        dy = sobel(dem_data, axis=0, mode='constant')

        # Calculate the second derivatives using the Sobel filter
        dxx = sobel(dx, axis=1, mode='constant')
        dyy = sobel(dy, axis=0, mode='constant')

        # Calculate the curvature using the second derivatives
        curvature = dxx + dyy

        # Smooth the curvature using Gaussian filtering (optional)
        curvature = gaussian_filter(curvature, sigma)

    return curvature
  
def calculate_gradients(dem_file):
    with rasterio.open(dem_file) as dataset:
        # Read the DEM data as a numpy array
        dem_data = dataset.read(1)

        # Calculate the gradients along the North and East directions
        dy, dx = np.gradient(dem_data, dataset.res[0], dataset.res[1])

        # Calculate the Northness and Eastness
        northness = np.arctan(dy / np.sqrt(dx**2 + dy**2))
        eastness = np.arctan(dx / np.sqrt(dx**2 + dy**2))

    return northness, eastness
  
  
def geotiff_to_csv(geotiff_file, csv_file, column_name):
    # Open the GeoTIFF file
    with rasterio.open(geotiff_file) as dataset:
        # Get the pixel values as a 2D array
        data = dataset.read(1)

        # Get the geotransform to convert pixel coordinates to geographic coordinates
        transform = dataset.transform

        # Get the width and height of the GeoTIFF
        height, width = data.shape

        # Open the CSV file for writing
        with open(csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the CSV header
            csvwriter.writerow(['Latitude', 'Longitude', 'x', 'y', column_name])

            # Loop through each pixel and extract latitude, longitude, and image value
            for y in range(height):
                for x in range(width):
                    # Get the pixel value
                    image_value = data[y, x]

                    # Convert pixel coordinates to geographic coordinates
                    lon, lat = transform * (x, y)

                    # Write the data to the CSV file
                    csvwriter.writerow([lat, lon, x, y, image_value])

  
def read_elevation_data(file_path, result_dem_csv_path, result_dem_feature_csv_path):
    neighborhood_size=4
    df = pd.read_csv(file_path)
    
    dataset = rasterio.open(geotiff_file)
    data = dataset.read(1)

    # Get the width and height of the GeoTIFF
    height, width = data.shape
    
    # Create an empty DataFrame with column names
    columns = ['lat', 'lon', 'elevation', 'slope', 'aspect', 'curvature', 'northness', 'eastness']
    all_df = pd.DataFrame(columns=columns)
    
    all_df.to_csv(result_dem_feature_csv_path)
    print(f"DEM and other columns are saved to file {result_dem_feature_csv_path}")
    return all_df


  
# Usage example:
result_dem_csv_path = "/home/chetana/gridmet_test_run/dem_template.csv"
result_dem_feature_csv_path = "/home/chetana/gridmet_test_run/dem_all.csv"


dem_file = "/home/chetana/gridmet_test_run/dem_file.tif"
slope_file = '/home/chetana/gridmet_test_run/slope_file.tif'
aspect_file = '/home/chetana/gridmet_test_run/aspect_file.tif'
curvature_file = '/home/chetana/gridmet_test_run/curvature_file.tif'
northness_file = '/home/chetana/gridmet_test_run/northness_file.tif'
eastness_file = '/home/chetana/gridmet_test_run/eastness_file.tif'


slope, aspect = calculate_slope_aspect(dem_file)
curvature = calculate_curvature(dem_file)
northness, eastness = calculate_gradients(dem_file)

# Save the slope and aspect as new GeoTIFF files
save_as_geotiff(slope, slope_file, dem_file)
save_as_geotiff(aspect, aspect_file, dem_file)
save_as_geotiff(curvature, curvature_file, dem_file)
save_as_geotiff(northness, northness_file, dem_file)
save_as_geotiff(eastness, eastness_file, dem_file)

geotiff_to_csv(dem_file, dem_file+".csv", "Elevation")
geotiff_to_csv(slope_file, slope_file+".csv", "Slope")
geotiff_to_csv(aspect_file, aspect_file+".csv", "Aspect")
geotiff_to_csv(curvature_file, curvature_file+".csv", "Curvature")
geotiff_to_csv(northness_file, northness_file+".csv", "Northness")
geotiff_to_csv(eastness_file, eastness_file+".csv", "Eastness")

# List of file paths for the CSV files
csv_files = [dem_file+".csv", slope_file+".csv", aspect_file+".csv", 
             curvature_file+".csv", northness_file+".csv", eastness_file+".csv"]

# Initialize an empty list to store all dataframes
dfs = []

# Read each CSV file into separate dataframes
for file in csv_files:
    df = pd.read_csv(file, encoding='utf-8')
    dfs.append(df)

# Merge the dataframes based on the latitude and longitude columns
merged_df = dfs[0]  # Start with the first dataframe
for i in range(1, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on=['Latitude', 'Longitude', 'x', 'y'])

# Save the merged dataframe to a new CSV file
merged_df.to_csv(result_dem_feature_csv_path, index=False)

