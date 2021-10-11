import numpy as np
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(x)
y = np.array([[1,2,3],[4,5,6],[7,8,9]]).T
print(y)
s = np.shape(x)
print(s)
w = np.array([5,2,4,1,8,7,9])
print(w[1])
print(w[3])
print(w[6])
print(np.shape(w))
e = np.array([[5,2,4,1,8,7,9]]).T
print(e)
print(np.shape(e))
r = np.zeros((7,7))
print(r)
t = np.ones((3,3))
print(t)
print(x[:,0])
print(x[0,:])

