getwd()
setwd("C:/Users/chris/Desktop/20BCAB23")
car=read.csv("CarPrice_Assignment.csv")
mean(car$horsepower)
# H0: Mean Horsepower = 104.1171
# H1: Mean Horsepower > 104.1171
t.test(car$horsepower,mu=104.1171,alternative ='two.sided',conf.level=0.95)
