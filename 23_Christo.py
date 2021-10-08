# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:44:48 2021

@author: chris
"""

import numpy as np
# Create and print 7 tuple 1D array
x=([1,2,3,4,5,6,7])
print(x)

#Print 2nd , 4th , 7th elements
print(x[1])
print(x[3])
print(x[6])

#Print the shape
print(np.shape(x))

#Print the transpose
X=np.array([[1,2,3,4,5,6,7]]).T
print(X)

#Create a 4x5 vector matrix
z=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,0,1]])
print(z)

#Print shape of avobe matrix
print(np.shape(z))

#Print the transpose of the matrix
Z=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,0,1]]).T
print(Z)

#8
print(z[:,0])

#9
print(z[0,:])

#Create a 7 dimensional array
print(np.zeros(7))
#Create a 3 dimensional array
print(np.array([[1,1,1]]*3))



