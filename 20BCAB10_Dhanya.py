# -*- coding: utf-8 -*-

"""

#1
import numpy as np
a=np.array([1,2,3,4,5,6,7])
#2
print(a[1])
print(a[3])
print(a[6])
#3
print(np.shape(a))
#4
a=np.array([[1,2,3,4,5,6,7]])
print(a)
b=np.array([[1,2,3,4,5,6,7]]).T
print(b)
#5
m=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[1,4,2,3,5]])
print(m)
#6
print(np.shape(m))
#7
print((m).T)
#8
print(m[:,0])
#9
print(m[0,:])
#10
print(np.zeros(7))
print(np.ones(3))
