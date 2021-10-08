# -*- coding: utf-8 -*-
"""
Created on fri Oct  8 10:15:38 2021

@author: dan christopher
"""

import numpy as np
#1

x=np.array([5,2,3,6,7,4,9])
print(x)

#2
print(x[1]) 
print (x[3])
print(x[6])

#3
print (np.shape(x))

#4
X=np.array((4,5,6,8,9,2,1)).T
print (X)

#5
z=np.array([[5,2,7,6,3],[14,23,16,28,56],[45,7,5,45,8],[5,4,22,36,8]])
print(z)

#6 
print(np.shape(z))

#7
Z=np.array([[5,2,7,6,3],[14,23,16,28,56],[45,7,5,45,8],[5,4,22,36,8]]).T
print(Z)

#8
print (z[:,0])

#9
print (z[0,:1])

#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))
