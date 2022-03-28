# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:53:35 2022

@author: Shahid
"""

import numpy as np
x= np.array([1,3,5,7,9,11,13])
print(x[1])
print(x[3])
print(x[6])
x = np.array([[1,2,3,4,5]])
print((x).T)
print(np.shape(x))
matrix = np.array([[1,2,3,4,5],[5,4,3,2,1],[6,7,8,9,10],[10,9,8,7,6]])
print(matrix)
print(matrix[0])
print(np.shape(matrix))
print((matrix).T)
print(np.shape((x).T))
print(matrix[0])
print(matrix[:,1])
print(np.zeros(7))
print(np.ones(3))
