# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:35:54 2021

@author: FIONA
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
X=np.array((3,5,8,1,4,9,78)).T
print(X)

#5
z=np.array([[11,41,2,30,12],[9,3,5,11,58],[3,7,88,46,81],[8,92,81,40,7]])
print(z)

#6
print(np.shape(z))

#7
Z=np.array([[11,41,2,30,12],[9,3,5,11,58],[3,7,88,46,81],[8,92,81,40,7]]).T
print(Z)

#8
print(z[:,0])

#9
print(z[0,:1])

#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))
