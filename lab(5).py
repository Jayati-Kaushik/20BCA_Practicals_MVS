# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:03:52 2022

@author: admin
"""

#hypothesis testing
setwd("C:/Users/admin/Desktop/Datasets")
data1=read.csv("CarPrice_Assignment.csv")
View(data1)
mean(data1$carlength)
#Ho: mean of carlength is equal to 170
#H1: mean of carlength is not equal to 170
t.test(data1$carlength,mu=170,alternative="less",conf.level=0.95)