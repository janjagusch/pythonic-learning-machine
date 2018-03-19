# Breast Cancer Wisconsin (Diagnostic) Data Set
# http://archive.ics.uci.edu/ml/data_sets/Breast+Cancer+Wisconsin+%28Diagnostic%29


require(magrittr)
require(readr)

# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_csv("01_raw/c_cancer/data_set.csv", 
                     col_names = FALSE)

# Removing ID variable.
data_set$X1 <- NULL

# Convert target variable.
target <- data_set$X2 %>% as.factor()
levels(target) <- c(0, 1)
target <- target %>% as.character() %>% as.numeric()

# Remove target from data set.
data_set$X2 <- NULL

# Append target at the end of data set.
data_set <- data.frame(data_set, target)

write.table(data_set, file = "02_cleaned//c_cancer.csv", row.names = F, col.names = F, sep = ",")
