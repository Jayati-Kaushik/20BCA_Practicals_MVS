#20BCAB42_Khushi 
import numpy as np
#x=np.array([[1,2,3],[3,2,1]])
#y=np.array([[1,2,3],[3,2,1]]).T
#print(x)
#print(y)

#7 tuple 1d array
a= np.array((1,2,3,4,5,6,7))
print(a)

#2,4,7th array element
print(a[1])
print(a[3])
print(a[6])

#array shape
print(a.shape)

#array transpose 
t=np.array([[1,2,3,4,5,6,7]]).T
print(t)
print(t.shape)

#4x5 matrix
m=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])
print(m)

#matrix shape
print(np.shape(m))

#matrix transpose 
mt=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]).T
print(mt)

#matrix first col
print(m[:,0])
#matrix first row
print(m[0,:])

#7D array with 0s, 3d array with 1s
print(np.zeros(7))
print(np.ones(3)) 
