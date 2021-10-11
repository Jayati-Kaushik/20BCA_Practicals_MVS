# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:13:15 2021

@author: prajw
"""

import numpy as np


y=(1,3,4,5,6,7,8)
print(y)
a=np.array([2,3,4,5,8,1,9])
print(a[1])
print(a[3])
print(a[6])
x = np.array([[1,2,6],[3,2,1],[9,12,7]])
print(x)
print(np.shape(y))
b= np.array([[1,2,6],[3,2,1],[9,12,7]]).T
print(b)
#c = np.array[[1,3,4],[5,3,6],[6,2,7],[2,13,16],[8,9,4]]
#print(c)
c = np.array([[1,2,6,7,3],[3,2,1,8,12],[9,12,7,4,8],[4,5,6,9,6]])
print(c)
print(np.shape(c))
#print first row of the matrix c
print(c[0])
#print first column of the matrix c
print(c[:,0:1])
#Transpose of the matrix c
d=np.array([[1,2,6,7,3],[3,2,1,8,12],[9,12,7,4,8],[4,5,6,9,6]]).T
print(d)
p=np.zeros((7,4))
print(p)
