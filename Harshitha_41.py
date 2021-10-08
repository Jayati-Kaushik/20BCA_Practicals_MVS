# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:31:15 2021

@author: Harshitha
"""

import numpy as np
#1
X=np.array((1,2,3,4,5,6,7,8,9))
print(X)
#2
print(X[2])
print(X[4])
print(X[7])
#3
print(np.shape(X))
#4
x=np.array([[1,2,3,4,5,6,7,8,9]]).T
print(x)
print(np.shape(x))
#5
y=np.array([[1,2,3,4,5],[3,4,5,6,7],[4,5,6,7,8],[8,9,5,6,7,]])
print(y)
#6
print(np.shape(y))
#7
Y=np.array([[1,2,3,4,5],[3,4,5,6,7],[4,5,6,7,8],[8,9,5,6,7,]]).T
print(Y)
#8
print(Y[1,1])
#9
print(Y[0,1])
#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1,]]*3))


