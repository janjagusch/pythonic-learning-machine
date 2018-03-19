# Geographical Original of Music Data Set 
# http://archive.ics.uci.edu/ml/datasets/geographical+original+of+music

require(magrittr)
require(readr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_csv("01_raw/r_music/data_set.csv", 
                     col_names = FALSE)

#Convert to numerical.
data_set$X27 <- data_set$X27 %>% as.character %>% as.numeric
data_set$X28 <- data_set$X28 %>% as.character %>% as.numeric

#Impute missing values.
data_set <- sapply(data_set, function(x){
  x[is.na(x)] <- median(x, na.rm = T)
  return(x)
}) %>% as.data.frame

#Remove one target variable.
data_set$X69 <- NULL

write.table(data_set, file = "02_cleaned//r_music.csv", row.names = F, col.names = F, sep = ",")
