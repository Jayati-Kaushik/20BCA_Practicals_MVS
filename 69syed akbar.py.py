# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#1
x=np.array([1,6,3,10,5,8,2])
print(x)

#2
print(x[1])
print(x[3])
print(x[6])

#3
print(np.shape(x))

#4
X=np.array((3,5,8,1,4,9,7)).T
print(X)

#5
z=np.array([[11,41,3,30,21],[9,4,6,11,45],[3,6,77,24,81],[8,92,81,40,5]])
print(z)

#6
print(np.shape(z))

#7
Z=np.array([[11,41,3,30,21],[9,4,6,11,45],[3,6,77,24,81],[8,92,81,40,5]]).T
print(Z)

#8
print(z[:,0])

#9
print(z[0,:1])

#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))





