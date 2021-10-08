# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:58:14 2021

@author: ROHIT
"""

import numpy as np


y=(1,2,3,4,5,6,7)
print(y)
a=np.array([2,3,4,5,8,1,9])
print(a[1])
print(a[3])
print(a[6])
x = np.array([[1,2,3],[3,2,1],[5,6,7]])
print(x)
print(np.shape(y))
b= np.array([[1,2,3],[3,2,1],[5,6,7]]).T
print(b)
#c = np.array[[1,3,4],[5,3,6],[6,2,7],[2,13,16],[8,9,4]]
#print(c)
c = np.array([[1,2,6,7,3],[3,2,1,8,12],[9,12,7,4,8],[4,5,6,9,6]])
print(c)
print(np.shape(c))
