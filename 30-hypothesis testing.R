car=read.csv("CarPrice_Assignment.csv")
View(car)

#H0: mean = 30
#H1: mean is not equal to 30
attach(car)
View(car)
mean(car$price)
t.test(car$price,mu=14000,conf.level = 0.95)


mean(car$enginesize)
t.test(car$enginesize,mu=120, conf.level = 0.95)