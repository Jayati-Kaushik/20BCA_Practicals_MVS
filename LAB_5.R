getwd()
setwd("C:/Users/admin/Downloads")
data=read.csv("CAR.csv")
View("data")
#H0: mean of enginesize is equal to 120
#H1: mean of enginesize is not equal to 120
mean(data$enginesize)      #mean =126.9073
t.test(data$enginesize,mu=120,alternative = "less",conf.level = 0.95)
