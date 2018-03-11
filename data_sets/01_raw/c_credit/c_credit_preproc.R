setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//c_credit//dataset.csv", sep = ",", header = F)

require(magrittr)

#Changing labels of target
dataset$V25 <- dataset$V25 - 1

write.table(dataset, file = "02_preproc//c_credit//dataset.csv", row.names = F, col.names = F, sep = ",")
