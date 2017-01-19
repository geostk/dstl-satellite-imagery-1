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

setwd('~/dev/competitions/kaggle/dstl-satellite-imagery/R')

library(raster)      # to read in from .tiff
library(rgdal)       # to read in from wkt
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

# dataframe to map the many polygon types to 10 classes
filename_to_classType = as.data.frame(matrix(c(
  '001_MM_L2_LARGE_BUILDING',1,
  '001_MM_L3_RESIDENTIAL_BUILDING',1,
  '001_MM_L3_NON_RESIDENTIAL_BUILDING',1,
  '001_MM_L5_MISC_SMALL_STRUCTURE',2,
  '002_TR_L3_GOOD_ROADS',3,
  '002_TR_L4_POOR_DIRT_CART_TRACK',4,
  '002_TR_L6_FOOTPATH_TRAIL',4,
  '006_VEG_L2_WOODLAND',5,
  '006_VEG_L3_HEDGEROWS',5,
  '006_VEG_L5_GROUP_TREES',5,
  '006_VEG_L5_STANDALONE_TREES',5,
  '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND',6,
  '007_AGR_L6_ROW_CROP',6, 
  '008_WTR_L3_WATERWAY',7,
  '008_WTR_L2_STANDING_WATER',8,
  '003_VH_L4_LARGE_VEHICLE',9,
  '003_VH_L5_SMALL_VEHICLE',10,
  '003_VH_L6_MOTORBIKE',10), ncol=2, byrow=T))
filename_to_classType$V1 = as.character(filename_to_classType$V1)
filename_to_classType$V2 = as.numeric(as.character(filename_to_classType$V2))

# read in the polygon grid sizes which will be needed to
# transform polys to same coords as rasters
grid_sizes = fread('../input/grid_sizes.csv')
setnames(grid_sizes, 'V1','ImageId')

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
  
  # then loop through the 10 other polygon classes
  # load and transform the polygons using affine trans
  # plot in overlay on raster
  
  for(i in 1:10) {
    
    class = filename_to_classType[filename_to_classType[,2]==i, 1]
    file00 = NULL
    
    for(cl in class) file00 = c(file00, dir(pattern=cl, path=paste0('../input/train_geojson_v3/', img_id), full.names = T))
    
    if(length(file00)!=0) {
      for(file0 in file00) {
        shp_poly <- readOGR(file0, 'OGRGeoJSON',require_geomType='wkbPolygon',  p4s="")
        plot(applyTransformation(at, shp_poly), add= TRUE, col=col0[i])
      }
    }
  }
  dev.off()
}

png('legend.png')
barplot(1:10,col=col0[1:10])
legend('topleft', legend=c('1 building','2 small struct', '3 good roads',
                           '4 track/trial','5 VEG','6 ARG','7 Waterway','8 water',
                           '9 large vh','10 small vh'),pch=15,col=col0[1:10])
dev.off()