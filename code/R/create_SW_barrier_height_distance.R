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
# rdemsw includes the larger extent that contains the costal line
rdemsw <- raster('D:/GoogleDrive/Research/P2_LRM_test/data/med/inputs/SNM_500_DEM_SW.tif')#'D:/GoogleDrive/Github/R/run_snm/input/dem/SNM_DEM_notscaled.tif'
rdemsw <- crop(rdemsw, c(extent(rdem)[1],
                         extent(rdemsw)[2],
                         extent(rdemsw)[3],
                         extent(rdemsw)[4]))

rw <- raster(paste0(Pathin,'SNM_wd2ocean.tif'))
rnw <-  raster(paste0(Pathin,'SNM_nwd2ocean.tif'))
rsw <- raster(paste0(Pathin,'SNM_swd2ocean.tif'))

vdem <- raster::as.matrix(rdem)
vdemsw <- raster::as.matrix(rdemsw)

vw <-  raster::as.matrix(rw)
vnw <- raster::as.matrix(rnw)
vsw <-  raster::as.matrix(rsw)

Nrow <- nrow(rdem)
Ncol <- ncol(rdem)

Nrowsw <- nrow(rdemsw)
Ncolsw <- ncol(rdemsw)

mwdis <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mwhei <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mnwdis <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mnwhei <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mswdis <- matrix(data=0, nrow = Nrow, ncol = Ncol)
mswhei <- matrix(data=0, nrow = Nrow, ncol = Ncol)

# south west  -------------------------------------------------------------------
# for(i in 2:Nrow){
#   for(j in 2:Ncol){
for(i in 1074:Nrow){
  for(j in 2:Ncol){
    if((Nrowsw-i+1) > j){
      suba <- vdemsw[i:(i+j-1),1:j]
      subsw <- vdemsw[i:(i+j-1),1:j]
      loc <- min((Nrowsw-i+1),j)*min((Nrowsw-i+1),j)-(j-1) # the loc index in subsw
    }else{
      suba <- vdemsw[i:Nrowsw,(j+i-Nrowsw):j]
      subsw <- vdemsw[i:Nrowsw,(j+i-Nrowsw):j]
      loc<- min((Nrowsw-i+1),j)*min((Nrowsw-i+1),j)-(Nrowsw-i)
    }
    # find peak location that closer to the pixel
    Peak <- which(revdiag(suba)==max(revdiag(suba),na.rm=TRUE))[length(which(revdiag(suba)==max(revdiag(suba),na.rm=TRUE)))]
    peakloc <- min((Nrowsw-i+1),j)*(Nrowsw-1)+(min((Nrowsw-i+1),j)-Nrowsw+1)
    
    #get barrier height
    if(loc == peakloc){
      mswdis[i,j] <- subsw[peakloc]
      mswhei[i,j] <- 0
      
    }else if(loc > peakloc){
      mswdis[i,j] <- subsw[loc]-  subsw[peakloc]# + dvocean[i,Peak]
      mswhei[i,j] <- suba[loc] - suba[peakloc]
      if(mswhei[i,j]>0){
        stop('wrong!')
      }
    }
    
  }
}


routswdis <- rdem
routswdis[] <- NA
values(routswdis) <- mswdis

routswhei <- rdem
routswhei[] <- NA
values(routswhei) <- mswhei

plot(routswdis)
plot(routswhei)


outname <- paste0(Pathout,'SNM_swbDistance_2.tif')
writeRaster(routswdis,outname, overwrite=T)
outname <- paste0(Pathout,'SNM_swbHeight_2.tif')
writeRaster(routswhei,outname, overwrite=T)

