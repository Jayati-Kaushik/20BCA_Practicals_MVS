# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:39:36 2022

@author: fiona
"""

getwd()
data=read.csv("CarPrice_Assignment.csv")
mean(data$curbweight)
# H0: Mean curbweight = 2550
# H1: Mean curbweight > 2550
t.test(data$curbweight,mu=2550,alternative ='two.sided',conf.level=0.95)