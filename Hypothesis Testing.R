getwd()
setwd("C:/Users/FIONA/Desktop/DATASETS")
car=read.csv("CarPrice_Assignment.csv")
mean(car$carheight)
# H0: Mean Height = 53.72488
# H1: Mean Height > 53.72488
t.test(car$carheight,mu=53.72488,alternative ='two.sided',conf.level=0.95)