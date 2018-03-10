setwd("C://Users//Jan//Documents//GitHub//TWEANN//data")

dataset <- read.csv("01_raw//r_music//dataset.csv", sep = ",", header = F)

#Convert to numerical
dataset$V27 <- dataset$V27 %>% as.character %>% as.numeric
dataset$V28 <- dataset$V28%>% as.character %>% as.numeric

#Impute missing values
dataset <- sapply(dataset, function(x){
  x[is.na(x)] <- median(x, na.rm = T)
  return(x)
}) %>% as.data.frame

#Remove one target variable
dataset$V69 <- NULL

write.table(dataset, file = "02_preproc//r_music//dataset.csv", row.names = F, col.names = F, sep = ",")
