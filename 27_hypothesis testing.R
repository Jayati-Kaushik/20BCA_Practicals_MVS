setwd("C:/Users/Dell/Downloads")
data=read.csv("CarPrice_Assignment.csv")
View(data)

attach(data)

mean(data$price)
t.test(data$price,mu=14000,conf.level=0.95)


mean(data$carlength)
t.test(data$carlength,mu=200,conf.level = 0.95)
