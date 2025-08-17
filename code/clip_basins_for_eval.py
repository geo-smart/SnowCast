
import os
import glob
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from snowcast_utils import data_dir, model_dir, output_dir

from rasterio.errors import WindowError


def clip_geotiffs_with_shapefile(folder_path, shapefile_path, name_pattern="*.tif", output_folder=None):
    shapefile = gpd.read_file(shapefile_path)

    if shapefile.shape[0] > 1:
        raise ValueError("Shapefile contains multiple polygons. Please use a shapefile with one polygon.")
    
    polygon = [shapefile.geometry.iloc[0].__geo_interface__]
    
    if output_folder is None:
        output_folder = folder_path
    
    os.makedirs(output_folder, exist_ok=True)

    geotiff_files = glob.glob(os.path.join(folder_path, name_pattern))

    for geotiff in geotiff_files:
        try:
            with rasterio.open(geotiff) as src:
                out_image, out_transform = mask(src, polygon, crop=True)
                
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                output_path = os.path.join(output_folder, f"clipped_{os.path.basename(geotiff)}")
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                print(f"Saved clipped file: {output_path}")
        except ValueError as e:
            print(f"Skipping {geotiff} due to error: {e}")
# Example usage:
# clip_geotiffs_with_shapefile("path/to/geotiffs", "path/to/shapefile.shp", name_pattern="*.tif", output_folder="path/to/output")


if __name__ == "__main__":
    clip_geotiffs_with_shapefile(
        folder_path = f"{output_dir}/", 
        shapefile_path = f"{data_dir}/shapefiles/TuolumneBasin.shp", 
        name_pattern="swe_predicted_*.tif", 
        output_folder=f"{output_dir}/TuolumneBasin_clipped/"
    )