#hypothesis testing
setwd("E:/JoelPaul/Desktop/Rstudios")
data=read.csv("CarPrice_Assignment.csv")
View(data)

# if p value is less than alpha(significance value)(alpha = 1-confidence level), we reject null hypthosis
#Ho: mean of enginesize = 120
#H1: mean of enginesize is not equal to 120
mean(data$enginesize) #mean = 126.9073
t.test(data$enginesize,mu=120,alternative="less",conf.level=0.95) #p=0.9908