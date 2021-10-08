# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#tuple
z=np.array([[1],[2],[3],[4],[5],[6],[7]])
print(z)
#transpose
t=np.array([[1,2,3,4,5,6,7]]).T
print(t)
#2nd element from array
print(t[2])
#4th element
print(t[4])
#6th element
print(t[6])
#shape of array
print(np.shape(t))
#vector 4x5
r=np.array([[1,2,3,7,4],[4,5,6,8,4],[7,8,9,5,0],[10,11,12,13,14]])
print(r)
#shape
print(np.shape(r))
#transpose
x=np.array([[1,2,3,7,4],[4,5,6,8,4],[7,8,9,5,0],[10,11,12,13,14]])
print(x)
#1st row
print(r[0,:])
#1st column
print(r[:,0])
#7 dimensional array with only 0s
s=np.zeros((7,7))
print(s)
#3 dimensional array with 1s
h=np.ones((3,3))
print(h)
