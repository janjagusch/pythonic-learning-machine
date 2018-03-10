setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//c_diabetes//dataset.csv", sep = ",", header = F)

write.table(dataset, file = "02_preproc//c_diabetes//dataset.csv", row.names = F, col.names = F, sep = ",")
