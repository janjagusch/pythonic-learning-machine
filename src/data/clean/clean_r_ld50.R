# Concrete

require(readr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_csv("01_raw/r_ld50/data_set.csv", 
                     col_names = FALSE)

# Export cleaned data.
write.table(data_set, file = "02_cleaned//r_ld50.csv", row.names = F, col.names = F, sep = ",")