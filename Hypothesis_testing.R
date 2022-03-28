setwd("C:/Users/Rose Maria Thomas/Desktop/joel")
c=read.csv("insurance.csv")
View(c)
summary(c)
mean(c$charges)
#H0: mean is 1500
#H1:mean is not 1500  
t.test(c$charges,mu=1500)
#H0:mean is 1500
#H1:mean is greater than 1500
t.test(c$charges,mu=1500,alternative = "greater",conf.level = 0.95)
#H0:mean charges on smokers = mean of charges on non smokers
#H1:mean charges on smokers != mean of charges on non smokers
t.test(c$charges~c$smoker,mu=0,alt='two.sided',conf=0.95,paired=FALSE)

