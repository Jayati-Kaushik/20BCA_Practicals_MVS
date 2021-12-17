setwd("C:/Users/prash/Downloads")
Car=read.csv("CarPrice_Assignment.csv")
View(Car)

#Ho: mean of enginesize = 120
#H1: mean enginesize is not equal to 120
mean(Car$enginesize)#mean=126.9073
t.test(Car$enginesize,mu=120,alternative = "less",conf.level =0.95) #0.9908


#Ho: mean of Horsepower =100
#H1: mean horsepower is not equal to 100
mean(Car$horsepower)
t.test(Car$horsepower,mu=100,alternative = "greater",conf.level = 0.98) #0.0688












































































# below is one tailed test whereas above one is two tailed
# perform one sample T test ( one tailed)
# Ho: mean is 30
# H1: mean is less than 30
t.test(uptake,mu=30,alternative="less",conf.level=0.95)
# Performs independent sample t test
# Ho: mean uptake of CO2 of quebec= mean uptake of CO2 of mississippi
# H1: mean uptake of CO2 of quebec!= mean uptake of CO2 of mississippi
t.test(uptake~Type,mu=0,alt='two.sided',conf=0.95,paired=FALSE)

