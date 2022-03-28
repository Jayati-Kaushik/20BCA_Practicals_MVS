# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:14:49 2021

@author: Acer
"""

import numpy as np
X=np.array([[1,2,3],[3,2,1],[9,10,7]])
print(X)
Y=np.array([[1,2,3],[3,2,1],[9,10,7]]).T
print(X)
A=np.array([[0]*3])
print(A)
print(np.zeros((4,8)))
print(np.ones(4))
print(X)
print(Y)
print(X+Y)
print(np.shape(X))
I=np.array([[10,20,30,40,50,60,70]])
print(I)
print(I[0,1],I[0,3],I[0,6])
print(np.shape(I))
print(I.T)
V=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(V)
print(np.shape(V))
print(V.T)
print(V[:,0])
print(V[0,:])
print(np.zeros(7))
print(np.ones(3))
