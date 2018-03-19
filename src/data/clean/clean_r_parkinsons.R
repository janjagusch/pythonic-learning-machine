# Parkinson Speech Dataset with Multiple Types of Sound Recordings Data Set 
# https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings

require(readr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_csv("01_raw/r_parkinsons/data_set.csv", 
                     col_names = FALSE)

#Remove subject id.
data_set$X1 <- NULL

#Remove classification label.
data_set$X29 <- NULL

# Export cleaned data.
write.table(data_set, file = "02_cleaned//r_parkinsons.csv", row.names = F, col.names = F, sep = ",")
