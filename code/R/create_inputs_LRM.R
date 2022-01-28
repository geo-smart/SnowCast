#create barrier heigh/distance, and distance to ocean. Only the direct from west is used.
# merge data into station_terrainData.csv and save the results into a new file station_terrainData_barrier.csv
# contact: Kehan Yang (kyang33@uw.edu)

library(raster)

data_dir <- "/Users/kehanyang/Documents/program/SnowCast/data/"
cop90_dem <- raster(paste0(data_dir, "cop90_dem/COP90_DEM_westernUS.tif")) 
station_data <- read.csv(paste0(data_dir, "station_gridcell/ground_measures_metadata.csv"))
station_topo_data <- read.csv(paste0(data_dir, "station_gridcell/station_terrainData.csv"))

# resample cop90_dem to 1000m --> as the satellite data all use WGS84 (geographic coordinate system), 
# and the study area covers a very wide range
# here I did not select projected coordinate system but simply resampled the dem
cop1k_dem = projectRaster(cop90_dem, res = res(cop90_dem)/90*1000, crs = crs(cop90_dem), method="bilinear")
# writeRaster(cop1k_dem, paste0(data_dir, "cop90_dem/cop1k_dem_westernUS.tif"))
cop1k_dem_mx = as.matrix(cop1k_dem)


df_results = data.frame()
for(i in 1:nrow(station_data)){
  
  station_coord <- cbind(station_data$longitude[i], station_data$latitude[i])
  id_cell = extract(cop1k_dem,SpatialPoints(station_coord), cellnumbers=TRUE)
  center_point = rowColFromCell(cop1k_dem, id_cell[1])
  # cells number = (row-1) * (column dimension) + col
  # get the strip from west
  dem_subset_west = cop1k_dem_mx[center_point[1], 1:center_point[2]]
  
  # plot(dem_subset_west)
  
  # W barrier height
  barrier_height_w = max(dem_subset_west, na.rm = T) - id_cell[2]
  # W barrier distance, (y_center - y_peak) *1000 to calculate the distance with unit km
  barrier_distance_w = (center_point[2] - which.max(dem_subset_west)) * 1000/1000

  # if there are >= 10 pixels (= 10 km) with 0 elevation, then the first pixel is seen as the shoreline (Ocean or big water body)
  N = 10
  for(j in center_point[2]:1){
    sl_subset = dem_subset_west[j:(j-9)]
    if(sum(sl_subset == 0) == N){
      # west distance to the ocean
      ocean_distance_w = center_point[2] - j
      break()
    }
  }
  
  temp_result = cbind.data.frame(station_id = station_data$station_id[i], barrier_heigh_west_m = barrier_height_w,
                               barrier_distance_west_km = barrier_distance_w, ocean_distance_west_km = ocean_distance_w)
  
  
  
  df_results = rbind(df_results, temp_result)
}

# merge data into the station_terrainData
merge_data = merge(station_topo_data, df_results, by = "station_id")
write.csv(merge_data, paste0(data_dir, "station_gridcell/station_terrainData_barrier.csv"), row.names = F)
