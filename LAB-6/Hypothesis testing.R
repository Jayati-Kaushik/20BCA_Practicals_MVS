getwd()
setwd("C:/Users/SONY/Documents/car data")
getwd()
car=read.csv("cars.csv")
View(car)
mean(car$wheelbase)
# H0: Mean wheelbase = 100
# H1: Mean wheelbase < 100
t.test(car$wheelbase,mu=100,alternative ='two.sided',conf.level=0.95)