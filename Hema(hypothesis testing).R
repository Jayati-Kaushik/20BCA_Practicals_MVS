#Hypothesis testing
setwd("C:/Users/S Hema/Desktop/R/r")
car=read.csv("CarPrice_Assignment.csv")
View(car)
#Ho:mean of enginesize=120
#H1:mean of enginesize is not equal to 120
mean(car$enginesize)  
t.test(car$enginesize,mu=120,alternative="less",conf.level =0.95 )
