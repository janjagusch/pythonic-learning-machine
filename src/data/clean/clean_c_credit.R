# Statlog (German Credit Data) Data Set 
# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

require(magrittr)
library(readr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load data set.
data_set <- read_csv("01_raw/c_credit/data_set.csv", 
                     col_names = FALSE)

#Changing labels of target.
data_set$X25 <- data_set$X25 - 1

# Export cleaned data set.
write.table(data_set, file = "02_cleaned//c_credit.csv", row.names = F, col.names = F, sep = ",")
