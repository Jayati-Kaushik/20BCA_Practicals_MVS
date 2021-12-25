getwd()
setwd("D:/NOTES/Datasets")
cars=read.csv("CarPrice_Assignment.csv")
View(cars)
mean(cars$citympg)
# H0: Mean of citympg = 25.21951
# H1: Mean of citympg > 25.21951
t.test(cars$citympg,mu=25.21951,alternative ='two.sided',conf.level=0.95)
