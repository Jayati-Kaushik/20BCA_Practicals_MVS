#20bcab59

setwd("C:/Users/Rabiya/Documents/BCADA")
data=read.csv(file.choose())
#if p value is less than alpha (significantvalue)(alpha=1-confidence level), we reject null hypothesis 
#H0
mean(data$carlength)
t.test(data$carlength,mu=120,conf.level = 0.95)

t.test(data$carlength,mu=120,alternative = "less",conf.level = 0.95)
