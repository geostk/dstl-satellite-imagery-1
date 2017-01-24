library(dplyr)

sample <- read.csv('~/dev/dstl-satellite-imagery/input/sample_submission.csv', stringsAsFactors = F)

sample <- mutate(sample, order = seq_along(ImageId))

predictions <- read.csv('~/dev/dstl-satellite-imagery/output/temp_logistic_buildings.csv', stringsAsFactors = F)

combined <- merge(sample, predictions, by = c('ImageId', 'ClassType'), all.x = T)
combined <- mutate(combined, MultipolygonWKT = ifelse(is.na(MultipolygonWKT.y), MultipolygonWKT.x, MultipolygonWKT.y)) %>%
            arrange(order) %>%
            select(ImageId, ClassType, MultipolygonWKT)

combined[combined$MultipolygonWKT == 'GEOMETRYCOLLECTION EMPTY',]$MultipolygonWKT <- 'MULTIPOLYGON EMPTY'

write.csv(combined, '~/dev/dstl-satellite-imagery/output/logistic_buildings.csv', row.names = F)
