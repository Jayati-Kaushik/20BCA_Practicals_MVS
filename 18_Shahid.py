# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:07:29 2021

@author: Shahid
"""

import numpy as np

#1
x=np.array([1,2,3,4,5,6,7])
print(x)

#2
print(x[1])
print(x[3])
print(x[5])

#3
print(np.shape(x))

#4
X=np.array((3,5,8,1,4,9,12)).T
print(X)

#5
z=np.array([[11,21,31,41,51],[9,8,7,6,5],[3,6,12,4,2],[1,5,9,2,6]])
print(z)

#6
print(np.shape(z))

#7
Z=np.array([[11,12,13,14,15],[1,2,13,14,15],[3,7,13,12,11],[3,4,5,6,7]])
print(Z)

#8
print(z[:,0])

#9
print(z[0,:1])

#10
print(np.array([[0,0,0,0,0,0,0,]]*7))
print(np.array([[1,1,1]]*3))
