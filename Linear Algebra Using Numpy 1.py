import numpy as np
# create and print 7 tuple 1D array
x=(11,34,56,23,50,7,10)
print(x)
# print 2nd,4th and 7th elements of array
#print 2nd element
print(x[1])
#print 4th element
print(x[3])
#print 7th element
print(x[6])
# print shape of an array
print(np.shape(x))
#print the transpose of array
X=np.array([[11,34,56,23,50,7,10]]).T
print(X)
print(np.shape(X))
#create 4x5 vector matrix
Y= np.array([[1,23,0,3,4],[1,1,4,7,0],[0,9,5,6,25],[23,7,8,0,2]])
print(Y)
#print the shape of the matrix
print(np.shape(Y))
#print the transpose of the matrix
Z=np.array([[1,23,0,3,4],[1,1,4,7,0],[0,9,5,6,25],[23,7,8,0,2]]).T
print(Z)
#print the first coloumn of matrix
print(Z[:,0])
#print the first row of the matrix
print(Z[0])
#create 7 dimensional array with only 0s,create a 3 dimensional array with only 1s
print(np.array([[0,0,0,0,0,0,0]]*7))
print(np.array([[1,1,1]]*3))