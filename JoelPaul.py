# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:52:22 2021

@author: joelp
"""
import numpy as np

x=(2,5,6,7,8,10,22)
print(x)

print(x[2])
print(x[5])
print(x[1])

print(np.shape(x))

X=np.array([[2,5,6,7,8,10,22]])
print(X)
print(np.shape(X))

z=np.array([[1,4,5,7,9],[27,36,12,44,12],[14,24,36,12,99],[19,76,54,66,88]])
print(z)

print(np.shape(z))

Z=np.array([[1,4,5,7,9],[27,36,12,44,12],[14,24,36,12,99],[19,76,54,66,88]]).T
print(Z)

print("First Column")
print(Z[0:1])

print("First Row")
print(Z[:, 0:1])

print(np.shape(Z))
print(Z.T)
print(Z[:,0])
print(Z[0,:])
print(np.zeros(7))
print(np.ones(3))

