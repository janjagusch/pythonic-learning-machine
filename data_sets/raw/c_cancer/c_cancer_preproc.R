# Breast Cancer Wisconsin (Diagnostic) Data Set
# http://archive.ics.uci.edu/ml/data_sets/Breast+Cancer+Wisconsin+%28Diagnostic%29

require(magrittr)

setwd("C://Users//Jan//Documents//GitHub//pythonic-learning-machine//data")

data_set <- read.csv("raw//c_cancer//data_set.csv", sep = ",", header = F)

# Removing ID variable
data_set$V1 <- NULL

# Convert target variable
target <- data_set$V2
levels(target) <- c(0, 1)
target <- target %>% as.character() %>% as.numeric()

# Remove target from data set
data_set$V2 <- NULL

# Append target at the end of data set
data_set <- data.frame(data_set, target)
colnames(data_set) <- 1:ncol(data_set)

write.table(data_set, file = "cleaned//c_cancer.csv", row.names = F, col.names = F, sep = ",")
