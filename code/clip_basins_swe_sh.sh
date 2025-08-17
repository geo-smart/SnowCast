#!/bin/bash

# Define input directories
data_dir="/home/chetana/data"
output_dir="/home/chetana/data/output"

# Define file paths
shapefile="$data_dir/shapefiles/TuolumneBasin.shp"
geojson="$data_dir/shapefiles/skagitboundary.json"

# Define output directories for clipped rasters
output_shp_clipped="$output_dir/TuolumneBasin_clipped_shp"
output_geojson_clipped="$output_dir/skagitboundary_clipped_geojson"

# Create output directories if they do not exist
mkdir -p "$output_shp_clipped"
mkdir -p "$output_geojson_clipped"

# Get the projection of the shapefile
shapefile_crs=$(ogrinfo -al -so "$shapefile" | grep "EPSG" | awk '{print $NF}')
echo "Shapefile CRS: EPSG:$shapefile_crs"

# Get the projection of the GeoJSON (assuming it has a CRS defined)
geojson_crs=$(ogrinfo -al -so "$geojson" | grep "EPSG" | awk '{print $NF}')
echo "GeoJSON CRS: EPSG:$geojson_crs"

# Loop through all matching GeoTIFFs
for geotiff in "$output_dir"/swe_predicted_*.tif; do
    # Check if the file exists
    if [[ ! -f "$geotiff" ]]; then
        echo "No matching GeoTIFF files found. Skipping..."
        continue
    fi

    # Get the CRS of the GeoTIFF
    geotiff_crs=$(gdalinfo "$geotiff" | grep "EPSG" | awk '{print $NF}')
    echo "Processing $geotiff with CRS: EPSG:$geotiff_crs"

    ## --- Process Shapefile Clipping ---
    
    # If shapefile CRS differs from GeoTIFF CRS, reproject it
    if [[ "$shapefile_crs" != "$geotiff_crs" ]]; then
        echo "Reprojecting shapefile to match GeoTIFF CRS..."
        ogr2ogr -t_srs EPSG:4326 "$data_dir/shapefiles/TuolumneBasin_reprojected.shp" "$shapefile"
        shapefile_to_use="$data_dir/shapefiles/TuolumneBasin_reprojected.shp"
    else
        shapefile_to_use="$shapefile"
    fi

    # Define output file name
    output_shp_file="$output_shp_clipped/clipped_$(basename "$geotiff")"

    # Remove the existing target file if it exists
    if [[ -f "$output_shp_file" ]]; then
        echo "Removing existing file: $output_shp_file"
        rm "$output_shp_file"
    fi

    # Run gdalwarp to clip the raster using the shapefile
    gdalwarp -cutline "$shapefile_to_use" -crop_to_cutline -dstnodata -9999 -co COMPRESS=LZW "$geotiff" "$output_shp_file"

    echo "Saved shapefile-clipped file: $output_shp_file"

    ## --- Process GeoJSON Clipping ---

    # If GeoJSON CRS differs from GeoTIFF CRS, reproject it
    if [[ "$geojson_crs" != "$geotiff_crs" ]]; then
        echo "Reprojecting GeoJSON to match GeoTIFF CRS..."
        ogr2ogr -t_srs EPSG:4326 "$data_dir/shapefiles/skagitboundary_reprojected.geojson" "$geojson"
        geojson_to_use="$data_dir/shapefiles/skagitboundary_reprojected.geojson"
    else
        geojson_to_use="$geojson"
    fi

    # Define output file name
    output_geojson_file="$output_geojson_clipped/clipped_$(basename "$geotiff")"

    # Remove the existing target file if it exists
    if [[ -f "$output_geojson_file" ]]; then
        echo "Removing existing file: $output_geojson_file"
        rm "$output_geojson_file"
    fi

    # Run gdalwarp to clip the raster using the GeoJSON
    gdalwarp -cutline "$geojson_to_use" -crop_to_cutline -dstnodata -9999 -co COMPRESS=LZW "$geotiff" "$output_geojson_file"

    echo "Saved GeoJSON-clipped file: $output_geojson_file"

done

echo "Zipping both folders into one file for downloading"
zip -r "$output_dir/clipped_swe_maps_for_eval.zip" "$output_shp_clipped" "$output_geojson_clipped"

echo "Zip file is ready $output_dir/clipped_swe_maps_for_eval.zip"
