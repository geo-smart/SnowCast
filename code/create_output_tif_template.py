import os
import rasterio
from rasterio.transform import from_origin
import numpy as np

def create_western_us_geotiff():
    # Define the spatial extent of the western U.S. (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = -125, 25, -100, 49

    # Define the resolution in degrees (4km is approximately 0.036 degrees)
    resolution = 0.036

    # Calculate the image size (width and height in pixels) based on the spatial extent and resolution
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Create an empty 2D NumPy array with a single band to store the image data
    data = np.zeros((height, width), dtype=np.float32)
    
    # read the grid geometry file
    homedir = os.path.expanduser('~')
    print(homedir)

    # Define the output filename
    output_filename = f"{homedir}/western_us_geotiff_template.tif"

    # Create the GeoTIFF file and specify the metadata
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,  # Single band
        dtype=np.float32,
        crs='EPSG:4326',  # WGS84
        transform=from_origin(minx, maxy, resolution, resolution),
    ) as dst:
        # Write the data to the raster
        dst.write(data, 1)

if __name__ == "__main__":
    create_western_us_geotiff()



