# -*- coding: utf-8 -*-
"""
Created on thu Oct  7 10:15:38 2021

@author: veda kumar
"""

import numpy as np
#1

x=np.array([1,2,3,4,5,6,7])
print(x)

#2
print(x[1]) 
print (x[3])
print(x[6])

#3
print (np.shape(x))

#4
X=np.array((6,8,4,5,3,1,9)).T
print (X)

#5
z=np.array([[21,11,45,23,22],[8,7,22,5,25],[6,4,16,24,28],[4,1,8,5,6]])
print(z)

#6 
print(np.shape(z))

#7
Z=np.array([[21,11,45,23,22],[8,7,22,5,25],[6,4,16,24,28],[4,1,8,5,6]]).T
print (Z)

#8
print (z[:,0])

#9
print (z[0,:1])

#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))