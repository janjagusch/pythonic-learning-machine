# Connectionist Bench (Sonar, Mines vs. Rocks) Data Set 
# http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

require(magrittr)
require(readr)


# Get working directory from command line arguments.
args <- commandArgs(trailingOnly = TRUE)
working_directory = args[1]

# Set working directory.
setwd(working_directory)

# Load raw data.
data_set <- read_csv("01_raw/c_sonar/data_set.csv", 
                     col_names = FALSE)

# Change target labels.
data_set$X61 <- data_set$X61 %>% as.factor()
levels(data_set$X61) <- c(0, 1)
data_set$X61 <- data_set$X61 %>% as.character %>% as.integer

# Export cleaned data.
write.table(data_set, file = "02_cleaned//c_sonar.csv", row.names = F, col.names = F, sep = ",")
