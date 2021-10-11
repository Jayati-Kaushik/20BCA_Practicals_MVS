# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:38:49 2021

@author: jacob jochim
"""

import numpy as np

z=(2,4,6,8,10,12,14)
print(z)
r=np.array([3,6,9,12,15,18,21])
print(r[1])
print(r[3])
print(r[6])
x = np.array([[1,2,6],[3,2,1],[9,12,7]])
print(x)
print(np.shape(x))
s= np.array([[1,2,6],[3,2,1],[9,12,7]]).T
print(s)
t = np.array([[1,2,6,7,3],[3,2,1,8,12],[9,12,7,4,8],[4,5,6,9,6]])
print(t)
print(np.shape(t))
print(t[0])
print(t[:,0:1])
a=np.array([[1,2,3,4,5],[3,6,9,12,15],[9,12,7,4,8],[4,5,6,9,6]]).T
print(a)
n=np.zeros((7,4))
print(n)