getwd()
coffee=read.csv("Coffee_Cleaned.csv")
View(coffee)
mean(coffee$Profit)#mean is 61.09
#H0:mean of profit =60
#H1:mean of profit is not =60
t.test(coffee$Profit,mu=60,alternative ="less",conf.level=0.95)

