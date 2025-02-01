#!/bin/bash
# this script will reproject and resample the western US dem, clip it, to match the exact spatial extent and resolution as the template tif
source ~/.bashrc

export PROJ_LIB="/home/geo2021/anaconda3/share/proj/"

gdalinfo --version

# Change directory to the working directory
cd /media/volume1/swe/gridmet_test_run
echo ">> Changed directory to /media/volume1/swe/data/gridmet_test_run"

# Create a directory for the template shapefile
mkdir -p /media/volume1/swe/data/template_shp/
echo ">> Created directory for template shapefile at /media/volume1/swe/data/template_shp/"

# Copy the template GeoTIFF to the template shapefile directory
cp /media/volume1/swe/data/western_us_geotiff_template.tif /media/volume1/swe/data/template_shp/
echo ">> Copied template GeoTIFF to template_shp directory"

# Generate the template shapefile from the copied GeoTIFF
# Remove the existing shapefile if it exists
if [ -f /media/volume1/swe/data/template.shp ]; then
    echo ">> Existing shapefile found. Removing it..."
    rm /media/volume1/swe/data/template.*
fi

# Generate the shapefile with the desired projection
echo ">> Generating new shapefile with WGS84 projection..."
gdaltindex -t_srs EPSG:4326 /media/volume1/swe/data/template.shp /media/volume1/swe/data/template_shp/*.tif

echo ">> Shapefile generation complete."
echo ">> Generated template shapefile using gdaltindex"

echo ">> Check on the cutline shapefile:"
ogrinfo -al -so /media/volume1/swe/data/template.shp

echo ">> Check on the merged DEM"
ls /media/volume1/swe/data/srtm/merged_dem.tif -lhtra
gdalinfo /media/volume1/swe/data/srtm/merged_dem.tif

echo ">> Check on new translated DEM"
gdalinfo /media/volume1/swe/data/srtm/srtm_wgs84_4km.tif

echo ">> Check test window on the DEM"
gdal_translate -projwin -113 38 -111 36 -of GTiff \
  /media/volume1/swe/data/srtm/srtm_wgs84_4km.tif \
  /tmp/srtm_subwindow_2.tif
gdalinfo -stats /tmp/srtm_subwindow_2.tif

# Reproject and resample the DEM, clipping it to match the template shapefile
echo ">> Cut the DEM with the shapefile"
gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -tr 0.036 0.036 -cutline /media/volume1/swe/data/template.shp -crop_to_cutline -overwrite /media/volume1/swe/data/srtm/srtm_wgs84_4km.tif /media/volume1/swe/data/srtm/output_dem_4km_clipped.tif
echo ">> Reprojected and resampled the DEM to match template shapefile, output saved to output_dem_4km_clipped.tif"


echo ">> Display information about the clipped output DEM"
gdalinfo -stats /media/volume1/swe/data/srtm/output_dem_4km_clipped.tif
echo ">> Displayed information about the clipped output DEM"

cp /media/volume1/swe/data/srtm/output_dem_4km_clipped.tif /media/volume1/swe/data/srtm/dem_file.tif


