import numpy as np

t=(1,2,3,4,5,6,7,8)
print(t)
p=np.array([2,4,6,8,9,10,11])
print(p[1])
print(p[3])
print(p[6])
x = np.array([[3,8,5],[3,2,1],[9,11,13]])
print(x)
print(np.shape(t))
s= np.array([[1,2,6],[3,2,1],[9,12,7]]).T
print(s)
y = np.array([[1,2,3,4,5],[3,6,9,12,15],[9,12,7,4,8],[4,5,6,9,6]])
print(y)
print(np.shape(y))
print(y[0])
print(y[:,0:1])
a=np.array([[1,2,3,4,5],[3,6,9,12,15],[9,12,7,4,8],[4,5,6,9,6]]).T
print(a)
b=np.zeros((7,4))
print(b)
