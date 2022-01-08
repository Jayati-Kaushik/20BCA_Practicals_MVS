getwd()
setwd("C:/Users/thoma/Desktop/LAB")
carsprice=read.csv("CarPrice_Assignment.CSV")
View(carprice)

# if the p value is less than alpha(significance value)(alpha = 1-confidence level), we reject null hypothesis.
mean(carsprice$price) #13276.71
#Ho : mean of carprice = 13000
#H1 : mean of carprice is not equal to 13000
t.test(carsprice$price,mu=13000,conf.level = 0.95)
