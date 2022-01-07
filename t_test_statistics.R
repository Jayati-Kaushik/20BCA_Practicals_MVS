getwd()
setwd("C:/Users/prajw/OneDrive/Desktop/rstudio")
data=read.csv("CarPrice_Assignment.csv")
View(data)
head(data)
mean(data$horsepower)

t.test(data$horsepower,mu=100)
