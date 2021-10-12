# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:31:09 2021

@author: Raisa
"""
import numpy as np
#1)create and print 7 tuple 1 D array
x=(3,5,8,1,4,9,78)
print(x)
#2)print 2nd 4th 7th elements of your array
print(x[1])
print(x[3])
print(x[6])
#3)shape of array
print(np.shape(x))
#4)tranpose of array
X=np.array([[3,5,8,1,4,9,78]]).T
print(X)
print(np.shape(X))
#5)4x5 vector matrix
z=np.array([[1,1,2,3,2],[22,3,55,1,5],[3,7,8,66,8],[98,2,1,0,7]])
print(z)
#6)shape of array
print(np.shape(z))
#7)tranpose of array
Z=np.array([[1,1,2,3,2],[22,3,55,1,5],[3,7,8,66,8],[98,2,1,0,7]]).T
print(Z)
#8)first row of matrix
print(z[1,4])
print(z[0])
#9)first col of matrix
print(z[:,0])
#10)7D array with 0 and 3D array with 1
print(np.array([[[[[[[0,0,2,3]]]]]]]))
print(np.array([[[1,1,1]]]))

