# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:06:00 2021

@author: josha
"""






#Q1
import numpy as np
a=np.array([1,2,3,4,5,6,7])
#Q2
print(a[1])
print(a[3])
print(a[6])
#Q3
print(np.shape(a))
#Q4
a=np.array([[1,2,3,4,5,6,7]])
print(a)
b=np.array([[1,2,3,4,5,6,7]]).T
print(b)
#Q5
m=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[1,4,2,3,5]])
print(m)
#Q6
print(np.shape(m))
#Q7
print((m).T)
#Q8
print(m[:,0])
#Q9
print(m[0,:])
#Q10
print(np.zeros(7))
print(np.ones(3))
