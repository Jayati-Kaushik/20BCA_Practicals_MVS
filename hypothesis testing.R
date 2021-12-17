getwd()
cars=read.csv("C:/Users/chris/csv files/CarPrice_Assignment.csv")
View(cars)

attach(cars)

mean(cars$price)
t.test(cars$price,mu=20000,conf.level = 0.95)

mean(cars$wheelbase)
t.test(cars$wheelbase,mu=25000,conf.level = 0.99)
