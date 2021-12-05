import numpy as np

#1
x=np.array([1,2,3,4,5,6,7])
print(x)

#2
print(x[1])
print(x[3])
print(x[6])

#3
print(np.shape(x))

#4
X=np.array((3,5,8,1,4,9,8)).T
print(X)

#5
z=np.array([[1,1,1,0,1],[1,1,2,3,5],[1,7,8,6,0],[5,2,1,0,6]])
print(z)

#6
print(np.shape(z))

#7
Z=np.array([[1,1,1,0,1],[1,1,2,3,5],[1,7,8,6,0],[5,2,1,0,6]]).T
print(Z)

#8
print(z[:,0])

#9
print(z[0,:1])

#10
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))
