# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:16:30 2022

@author: tejas
"""

getwd()
setwd("C:/Users/tejas/Documents/jerry(DS)/datasets")
data=read.csv("heart.csv")
t.test(data$RestingBP,mu=30)
t.test(data$RestingBP,mu=132 ,alternative="less",conf.level=0.85)
mean(data$RestingBP)

#if p value less than alpha(significance value = 1 - confidence level) reject the null hyposthesis
#Ho: mean = 30
#H1: mean is not equal to 30
#attach(CO2)
#View(CO2)
#t.test(uptake,mu=30)# if mu=27.2131, p-value=1
# below is one tailed test whereas above one is two tailed
# perform one sample T test ( one tailed)
# Ho: mean is 30
# H1: mean is less than 30
#t.test(uptake,mu=30,alternative="less",conf.level=0.95)
# Performs independent sample t test
# Ho: mean uptake of CO2 of quebec= mean uptake of CO2 of mississippi
# H1: mean uptake of CO2 of quebec!= mean uptake of CO2 of mississippi
#t.test(uptake~Type,mu=0,alt='two.sided',conf=0.95,paired=FALSE)