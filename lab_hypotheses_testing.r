#hypothesis testing
setwd("C:/Users/Harshitha/Downloads")
car=read.csv("car_assignment.csv")

#Ho:mean of enginesize=120
#H1:mean of enginesize is not equal to 120
mean(data$enginesize)  #mean being 126.8971
t.test(data$enginesize,mu=120,alternative="less",conf.level =0.95 ) #we get p=0.9908
