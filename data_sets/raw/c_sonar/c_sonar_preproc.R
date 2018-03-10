setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//c_sonar//dataset.csv", sep = ",", header = F)

require(magrittr)

#Changing labels of target
levels(dataset$V61) <- c(0, 1)
dataset$V61 <- dataset$V61 %>% as.character %>% as.integer

write.table(dataset, file = "02_preproc//c_sonar//dataset.csv", row.names = F, col.names = F, sep = ",")
