# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#1
x=np.array([1,3,5,7,9,11,13])


#2
print(x[1])
print(x[3])
print(x[6])

#3
print(np.shape(x))

#4
x=np.array([[1,3,5,7,9,11,13]]).T
print(x)
print(np.shape(x))

#5
z=np.array([[1,1,2,3,2],[22,3,5,1,5],[2,7,8,6,3],[9,2,1,0,7]])
print(z)

#6
print(np.shape(z))

#7
z=np.array([[1,1,2,3,2],[2,3,5,1,5],[2,7,8,6,3],[9,2,1,0,7]]).T
print(z)

#8
print(z[0])

#9
print(z[:,0])

#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))