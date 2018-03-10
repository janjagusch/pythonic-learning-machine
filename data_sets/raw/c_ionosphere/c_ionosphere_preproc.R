setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//c_ionosphere//dataset.csv", sep = ",", header = F)

write.table(dataset, file = "02_preproc//c_ionosphere//dataset.csv", row.names = F, col.names = F, sep = ",")
