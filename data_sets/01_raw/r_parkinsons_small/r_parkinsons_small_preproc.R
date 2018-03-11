setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//r_parkinsons_small//dataset.csv", sep = ",", header = F)

#Removing subject id
dataset$V1 <- NULL

#Removing classification label
dataset$V29 <- NULL

write.table(dataset, file = "02_preproc//r_parkinsons_small//dataset.csv", row.names = F, col.names = F, sep = ",")
