setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//r_parkinsons//dataset.csv", sep = ",", header = T)

write.table(dataset, file = "02_preproc//r_parkinsons//dataset.csv", row.names = F, col.names = F, sep = ",")
