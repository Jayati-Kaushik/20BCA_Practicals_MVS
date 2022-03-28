getwd()
data=read.csv("CarPrice_Assignment.csv")
mean(data$curbweight)
 H0: Mean curbweight = 2550
 H1: Mean curbweight > 2550
t.test(data$curbweight,mu=2550,alternative ='two.sided',conf.level=0.95)


