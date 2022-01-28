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


# west  -------------------------------------------------------------------
for(i in 1:Nrow){
  a <- vdem[i,]
  for(j in 2:Ncol){
    suba <- a[1:j]
    # find peak location that closer to the pixel
    Peak <- which(suba==max(suba,na.rm=TRUE))[length(which(suba==max(suba,na.rm=TRUE)))]
    #get barrier height
    if(j == Peak){
      mwdis[i,j] <- vw[i,j]
      mwhei[i,j] <- 0
    }else if(j > Peak){
      mwdis[i,j] <- vw[i,Peak]- vw[i,j]# + dvocean[i,Peak]
      mwhei[i,j] <- vdem[i,j] - vdem[i,Peak]
    }
  }
}

routwdis <- rdem
routwdis[] <- NA
values(routwdis) <- mwdis

routwhei <- rdem
routwhei[] <- NA
values(routwhei) <- mwhei

plot(routwdis)
plot(routwhei)


outname <- paste0(Pathout,'SNM_wbDistance.tif')
writeRaster(routwdis,outname, overwrite=T)
outname <- paste0(Pathout,'SNM_wbHeight.tif')
writeRaster(routwhei,outname, overwrite=T)


