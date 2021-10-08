# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:12:30 2021

@author: asus
"""

import numpy as np
x=np.array([1,2,3,4,5,6,7])
print(x)
print(x[1],x[3],x[6])
print(np.shape(x))
y=x.T
print(y)
a=np.array([[1,2,3,4,5,],[5,6,7,8,9,],[9,8,7,6,5],[1,3,5,7,9]])
print(a)
print(np.shape(a))
b=a.T
print(b)
print(np.shape(b))
print(a[0,0],a[1,0],a[2,0],a[3,0])
print(a[0,0],a[0,1],a[0,2],a[0,3],a[0,4])
k=np.zeros((7))
print(k)
s=np.ones((3))
print(s)