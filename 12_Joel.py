# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:01:43 2021

@author: JOEL
"""

import numpy as np
#create and print 7 tuple ID array
x=([1,2,3,4,5,6,7])
print(x)

#print 2nd,4th,7th elements
print(x[1])
print(x[3])
print(x[6])

#print the shape
print(np.shape(x))

#print the transpose
X=np.array([[1,2,3,4,5,6,7]]).T
print(X)

#create a 4x4 vector matrix
z=np.array([[7,8,3,3,2],[19,3,22,1,5],[18,7,3,23,18],[2,5,10,9,7]])
print(z)

#print shape of avobe matrix
print(np.shape(z))

#print the transpose of the matix
Z=np.array([[7,8,3,3,2],[19,3,22,1,5],[18,7,3,23,18],[2,5,10,9,7]]).T
print(Z)

#print the first coloumn of the matix
print(z[:,0])

#print the first row of the matix
print(z[0,:1])

#create a 7 dimensional array
a=np.zeros((7))
print(a)
print(np.array([[1,1,1]]*3))


