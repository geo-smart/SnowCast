# create distance to barrier and height difference to barrier -------------
#resample dem to study area

library(raster)
library(StationSWERegressionV2)
library(geosphere)
library(lava)

Pathin <- '/scratch/summit/keya6539/LRM_test/input/Barrier/'
Pathin <- 'D:/GoogleDrive/Research/P2_LRM_test/data/template/'
Pathout <- Pathin
rdem <- raster(paste0(Pathin,'SNM_dem.tif'))#'D:/GoogleDrive/Github/R/run_snm/input/dem/SNM_DEM_notscaled.tif'


rw <- raster(paste0(Pathin,'SNM_wd2ocean.tif'))
rnw <-  raster(paste0(Pathin,'SNM_nwd2ocean.tif'))
rsw <- raster(paste0(Pathin,'SNM_swd2ocean.tif'))

vdem <- raster::as.matrix(rdem)

vw <-  raster::as.matrix(rw)
vnw <- raster::as.matrix(rnw)
vsw <-  raster::as.matrix(rsw)

Nrow <- nrow(rdem)
Ncol <- ncol(rdem)

mwdis <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mwhei <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mnwdis <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mnwhei <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mswdis <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mswhei <- matrix(data=0, nrow = Nrow, ncol = Ncol)


# north west  -------------------------------------------------------------------
for(i in 2:Nrow){
  for(j in 2:Ncol){
    if(i > j){
      suba <- vdem[(i-j+1):i,1:j]
      subnw <- vnw[(i-j+1):i,1:j]
    }else{
      suba <- vdem[1:i,(j-i+1):j]
      subnw <- vnw[1:i,(j-i+1):j]
    }
    # find peak location that closer to the pixel
    Peak <- which(diag(suba)==max(diag(suba),na.rm=TRUE))[length(which(diag(suba)==max(diag(suba),na.rm=TRUE)))]
    loc <- min(i,j)*min(i,j)
    peakloc <- min(i,j)*(Peak-1)+Peak
    #get barrier height
    # col first
    if(loc == peakloc){
      mnwdis[i,j] <- subnw[peakloc]
      mnwhei[i,j] <- 0
    }else if(loc > peakloc){
      mnwdis[i,j] <- subnw[loc]-  subnw[peakloc]# + dvocean[i,Peak]
      mnwhei[i,j] <- suba[loc] - suba[peakloc]
    }
  }
}

routnwdis <- rdem
routnwdis[] <- NA
values(routnwdis) <- mnwdis

routnwhei <- rdem
routnwhei[] <- NA
values(routnwhei) <- mnwhei

plot(routnwhei)
plot(routnwdis)

outname <- paste0(Pathout,'SNM_nwbHeight.tif')
writeRaster(routnwhei,outname, overwrite=T)
outname <- paste0(Pathout,'SNM_nwbDistance.tif')
writeRaster(routnwdis,outname, overwrite=T)


