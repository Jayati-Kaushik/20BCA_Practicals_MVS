# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:49:03 2021

@author: hp
"""

#1
import numpy as np
x=np.array([1,2,3,4,5,6,7])
#2
print(x[2])
print(x[4])
print(x[6])
#3

print(np.shape(x))
#4
x=np.array([[3,4,9,11,14,15]]).T
print(x)
print(np.shape(x))

#5
y=np.array([[1,2,3,4,5],[8,9,7,5,4],[8,5,2,3,1],[7,8,9,4,5]])
print(y)
#6
print(np.shape(y))
#7
y=np.array([[1,2,3,4,5],[8,9,7,5,4],[8,5,2,3,1],[7,8,9,4,5]]).T

print(y)
#8
print("first column")
print(y[:,0:1])
#9
print("first row ")
print(y[0])
