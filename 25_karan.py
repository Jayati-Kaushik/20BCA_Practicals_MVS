# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:12:41 2021

@author: Deepak SK
"""

import numpy as np
x= np.array([1,2,3,4,5,6,7])
print(x[1])
print(x[3])
print(x[6])
print(np.shape(x))
print(x)
y=np.array([[1,2,3,4,5,6,7]]).T
print(y)
print(np.shape(y))
a=np.array([[1,2,3,4],[9,8,7,6],[3,4,5,6],[6,2,6,4]])
print(a)
print(a[0,:])
print(a[:,0])
z=np.transpose(a)
print(z)
s=np.zeros((7,7))
print(s)
#3 dimensional array with 1s
h=np.ones((3,3))
print(h)