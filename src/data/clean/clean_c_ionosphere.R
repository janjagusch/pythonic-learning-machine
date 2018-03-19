# Ionosphere Data Set 
# https://archive.ics.uci.edu/ml/datasets/ionosphere

require(readr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_csv("01_raw/c_ionosphere/data_set.csv", 
                     col_names = FALSE)

# Export cleaned data.
write.table(data_set, file = "02_cleaned//c_ionosphere.csv", row.names = F, col.names = F, sep = ",")
