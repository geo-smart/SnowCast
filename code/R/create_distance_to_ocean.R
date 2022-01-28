
library(raster)
library(StationSWERegressionV2)
library(geosphere)
library(lava)
# source('D:/GoogleDrive/Github/R/StationSWERegressionV2/R/Ffill_NA.R')

# read coaste line raster -------------------------------------------------
rdem <- raster('D:/GoogleDrive/Research/P2_LRM_test/data/template/SNM_500_DEM.tif')
r_shoreline_arcgis <- raster('D:/GoogleDrive/Research/P2_LRM_test/data/template/California_shoreline_simplified_raster4.tif')
r_shoreline_arcgis[r_shoreline_arcgis==0] <- 1
# writeRaster(r_shoreline_arcgis,'D:/GoogleDrive/Research/P2_LRM_test/data/template/California_shoreline_simplified_raster5.tif')
# plot(r_shoreline_arcgis)
r_shoreline_crop <- crop(r_shoreline_arcgis,extent( extent(r_shoreline_arcgis)[1],
                                                    extent(rdem)[2], 
                                                    max(extent(rdem)[3]-15/3600*nrow(rdem)/2, extent(r_shoreline_arcgis)[3]), 
                                                    min(extent(rdem)[4]+15/3600*nrow(rdem),extent(r_shoreline_arcgis)[4])))


# the min lat of r_shoreline_arcgis is larger than extent(rdem)[3]-15/3600*nrow(rdem)/2
# the max lon of r_shoreline_arcgis is smaller than extent(rdem)[4]+15/3600*nrow(rdem)
r_shoreline_crop[!is.na(r_shoreline_crop)] <- 1
plot(r_shoreline_crop)
writeRaster(r_shoreline_crop,'D:/GoogleDrive/Research/P2_LRM_test/data/template/California_shoreline_simplified_raster_crop5.tif')
dfshoreline <- as.matrix(r_shoreline_crop)

Pathout <- 'D:/GoogleDrive/Research/P2_LRM_test/data/template/'





# create distance to ocean  -----------------------------------------------

if(T){
  # calculate west distance to ocean 
  temp <- raster('D:/GoogleDrive/Research/P2_LRM_test/data/template/SNM_500_DEM.tif')
  p_temp <- rasterToPoints(temp)
  wd2ocean <- p_temp
  head(p_temp)
  for(i in 1:nrow(temp)){
    # the min lat of r_shoreline_arcgis is larger than extent(rdem)[3]-15/3600*nrow(rdem)/2
    # top half, and botton is less than half
    m = as.integer((extent(r_shoreline_crop)[4]-extent(temp)[4])/(15/3600) +i) # the related location in dfshoreline
    loc <- which(dfshoreline[m,] == 1)
    loc <- max(loc)
    for(j in 1:ncol(r_shoreline_crop)){
      # dis <-  ncol(r_shoreline_crop)-ncol(temp)+ j - loc # grid numbers
      
      Lat <- extent(r_shoreline_crop)[4] - 15/3600*(m-1)
      Lon <- extent(r_shoreline_crop)[1] + 15/3600*(loc-1)
      
      index <- (i-1)*ncol(temp)+j # index of distance
      # 1 -- lon, 2 -- lat, 3 -- values 
      wd2ocean[index,3] <- distm(c(wd2ocean[index,1], wd2ocean[index,2]), c(Lon, Lat), fun = distHaversine)/1000 # unit: km
    }
    
  }
  
  
  # read distance to ocean tempolate ----------------------------------------
  nwd2ocean <- p_temp
  swd2ocean <- p_temp
  
  # loop to calculate northwest and south west distance to ocean ------------
  
  for(i in 1478:nrow(temp)){
    for(j in 1:ncol(temp)){
      # for the northwest distance to the ocean
      m <-  as.integer((extent(r_shoreline_crop)[4]-extent(temp)[4])/(15/3600) +i)
      n <-  as.integer((extent(temp)[1]-extent(r_shoreline_crop)[1])/(15/3600)+ j)
      
      while (m <= nrow(dfshoreline) && n <= ncol(dfshoreline) && m >0 && n >0 && is.na(dfshoreline[m,n])) {
        m = m-1
        n = n-1
      }
      
      if(m==0 || n==0){
        index <- (i-1)*ncol(temp)+j
        nwd2ocean[index,3] <- NA
        
      }else{
        # Lat <- 47.62662 - 0.004166667*m
        # Lon <- -124.9984 + 0.004166667*n
        
        Lat <- extent(r_shoreline_crop)[4] - 15/3600*(m-1)
        Lon <- extent(r_shoreline_crop)[1] + 15/3600*(n-1)
        
        index <- (i-1)*ncol(temp)+j
        nwd2ocean[index,3] <- distm(c(nwd2ocean[index,1], nwd2ocean[index,2]), c(Lon, Lat), fun = distHaversine)/1000
      }
      
      # for the sorthwest distance to the ocean
      m <-  as.integer((extent(r_shoreline_crop)[4]-extent(temp)[4])/(15/3600) +i)
      n <-  as.integer((extent(temp)[1]-extent(r_shoreline_crop)[1])/(15/3600)+ j)
      
      while (m >0 && n >0 && m <= nrow(dfshoreline) && is.na(dfshoreline[m,n])){
        m <- m + 1
        n <- n - 1
        
      }
      
      if(m>nrow(r_shoreline_crop) || n==0){
        index <- (i-1)*ncol(temp)+j
        swd2ocean[index,3] <- NA
        
      }else{
        # Lat <- 47.62662 - 0.004166667*m
        # Lon <- -124.9984 + 0.004166667*n
        # 
        Lat <- extent(r_shoreline_crop)[4] - 15/3600*(m-1)
        Lon <- extent(r_shoreline_crop)[1] + 15/3600*(n-1)
        
        index <- (i-1)*ncol(temp)+j
        swd2ocean[index,3] <- distm(c(swd2ocean[index,1], swd2ocean[index,2]), c(Lon, Lat), fun = distHaversine)/1000
      }
      
    }
  }
  
  
  # setvalues to raster -----------------------------------------------------
  # there is an error here
  rwd2ocean <- temp
  rnwd2ocean <- temp
  rswd2ocean <- temp
  values(rwd2ocean)<- wd2ocean[,3]
  values(rnwd2ocean)<- nwd2ocean[,3]
  values(rswd2ocean) <- swd2ocean[,3]
  
  rwd2ocean[is.na(rswd2ocean)] <- NA
  rwd2ocean[is.na(rnwd2ocean)] <- NA
  rswd2ocean[is.na(rwd2ocean)] <- NA
  rnwd2ocean[is.na(rwd2ocean)] <- NA
  
  rwd2ocean[is.na(rwd2ocean)] <- 0
  rswd2ocean[is.na(rswd2ocean)] <- 0
  rnwd2ocean[is.na(rnwd2ocean)] <- 0
  
  writeRaster(rwd2ocean,'D:/GoogleDrive/Research/P2_LRM_test/data/template/SNM_wd2ocean.tif')
  writeRaster(rnwd2ocean,'D:/GoogleDrive/Research/P2_LRM_test/data/template/SNM_nwd2ocean.tif')
  writeRaster(rswd2ocean,'D:/GoogleDrive/Research/P2_LRM_test/data/template/SNM_swd2ocean.tif')
}


