setwd("C://Users//Jan//Documents//GitHub//TWEANN//data//01_raw//r_student")

dataset <- read.table("student-mat.csv",sep=";",header=TRUE)

#Convert int to factor
dataset$Medu <- dataset$Medu  %>% as.factor
dataset$Fedu <- dataset$Fedu %>% as.factor

dataset$traveltime <- dataset$traveltime %>% as.factor
dataset$studytime <- dataset$studytime %>% as.factor
dataset$failures <- dataset$failures %>% as.factor

#Remove two target variables
dataset$G1 <- NULL
dataset$G2 <- NULL

require(dummies)

dataset <- dummy.data.frame(dataset)

setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

write.table(dataset, file = "02_preproc//r_student//dataset.csv", row.names = F, col.names = F, sep = ",")
