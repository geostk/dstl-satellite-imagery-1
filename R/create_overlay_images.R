#  Nigel Carpenter Jan 2017
#
#  Reworking of Kernel by Randel to overlay polygons on raster images.
#  Original Kernel can be found here
#  https://www.kaggle.com/randel/dstl-satellite-imagery-feature-detection/25-images-and-polygons-side-by-side-eda/code
#
#  This script could be adapted to save transformed polys to same coords as rasters 
#  from which could then apply land use classification techniques like those in QGis
#
#  This script reads in the 3 band images and the polygons to create overlay images

setwd('~/dev/dstl-satellite-imagery/R')

library(raster)      # to read in from .tiff
library(rgdal)       # to read in from wkt
library(rgeos)
library(geojsonio)   # to read in from geojson

library(data.table)  # fast csv read
library(vec2dtransf) # for polygon to raster registration using affine transformation

# Input data files are available in the "../input/" directory.

# list of all the training images
wkt = fread('../input/train_wkt_v4.csv')

# change 'MULTIPOLYGON EMPTY' to NA
wkt[wkt$MultipolygonWKT=='MULTIPOLYGON EMPTY','MultipolygonWKT'] = NA

# define a colour pallette for polygon overlays
col0 = rainbow(n=10, alpha = .5)

predictions <- read.csv('~/dev/dstl-satellite-imagery/output/temp_xgboost_buildings_all_train_scaled_patches_256_3_3.csv', stringsAsFactors = F)

# read in the polygon grid sizes which will be needed to
# transform polys to same coords as rasters
grid_sizes = fread('../input/grid_sizes.csv')
setnames(grid_sizes, 'V1','ImageId')

img_id <- '6120_2_2'
for(img_id in unique(wkt$ImageId)) {
  
  print(img_id)
  png(paste0('../images/', img_id,".png"), height=720, width=720)
  
  #read X & Y extents for polygon Vector file
  x_Vmax = grid_sizes[grid_sizes$ImageId==img_id]$Xmax
  y_Vmin = grid_sizes[grid_sizes$ImageId==img_id]$Ymin
  x_Vmin = 0
  y_Vmax = 0
  
  #Load the raster image and get its X and Y extents
  img_raster <- stack(paste0("../input/three_band/", img_id, ".tif"))
  
  x_Rmax <- img_raster@extent[2]
  x_Rmin <- img_raster@extent[1]
  y_Rmax <- img_raster@extent[4]
  y_Rmin <- img_raster@extent[3]
  
  # define control points that will be used to transform polygons
  # onto same coords as raster using affine trans from vec2dtransf package
  control.points <- data.frame("X source" = 0, 'Y source' = 0, 'X target' = 0, 'Y target' = 0)
  
  control.points[1,] <- unlist(c(x_Vmin, y_Vmin, x_Rmin, y_Rmin))
  control.points[2,] <- unlist(c(x_Vmax, y_Vmax, x_Rmax, y_Rmax))
  control.points[3,] <- unlist(c(x_Vmin, y_Vmax, x_Rmin, y_Rmax))
  control.points[4,] <- unlist(c(x_Vmax, y_Vmin, x_Rmax, y_Rmin))
  
  # calculate the parameters of the transformation that will be applied later
  at <- AffineTransformation(control.points)
  calculateParameters(at)
  
  # now plot the raster image
  plotRGB(img_raster, stretch = "lin")
  
  poly <- readWKT(predictions[predictions$ImageId==img_id & predictions$ClassType == 1,]$MultipolygonWKT)
  #poly <- readWKT(wkt[wkt$ImageId==img_id & wkt$ClassType == 1,]$MultipolygonWKT)
  
  plotRGB(img_raster, stretch = "lin")
  plot(applyTransformation(at, poly), add = TRUE, col=col0[1])
  
  # then loop through the 10 other polygon classes
  # load and transform the polygons using affine trans
  # plot in overlay on raster
  
  # for(i in 1:10) {
  #   
  #   class = filename_to_classType[filename_to_classType[,2]==i, 1]
  #   file00 = NULL
  #   
  #   for(cl in class) file00 = c(file00, dir(pattern=cl, path=paste0('../input/train_geojson_v3/', img_id), full.names = T))
  #   
  #   if(length(file00)!=0) {
  #     for(file0 in file00) {
  #       shp_poly <- readOGR(file0, 'OGRGeoJSON',require_geomType='wkbPolygon',  p4s="")
  #       plot(applyTransformation(at, shp_poly), add= TRUE, col=col0[i])
  #     }
  #   }
  # }
  # dev.off()
}





img_id <- '6120_2_2'
img_raster <- stack(paste0("../input/sixteen_band/", img_id, "_M.tif"))
img_raster <- stack(paste0("../input/three_band//", img_id, ".tif"))
plot(img_raster[[1]])
plotRGB(img_raster, stretch = "lin")
  
dev.off()