# Student Performance Data Set 
# https://archive.ics.uci.edu/ml/datasets/student+performance

require(magrittr)
require(readr)
require(dummies)
require(dplyr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_delim("01_raw/r_student/data_set.csv", 
                       ";", escape_double = FALSE, trim_ws = TRUE)

#Convert int to factor.
data_set$Medu <- data_set$Medu  %>% as.factor
data_set$Fedu <- data_set$Fedu %>% as.factor

data_set$traveltime <- data_set$traveltime %>% as.factor
data_set$studytime <- data_set$studytime %>% as.factor
data_set$failures <- data_set$failures %>% as.factor

#Remove two target variables.
data_set$G1 <- NULL
data_set$G2 <- NULL

# Convert character to factor.
cols <- sapply(data_set, is.character)
data_set[, cols] <- lapply(data_set[, cols], as.factor)

# Convert categorical to dummy numerical.
data_set <-  dummy.data.frame(data_set)

# Export cleaned data.
write.table(data_set, file = "r_student.csv", row.names = F, col.names = F, sep = ",")
