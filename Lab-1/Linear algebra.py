# 1. Creating and printing 7 tuples in the form of 1D array
import numpy as np
x=np.array([[1,2,3,4,5,6,7]])
print(x)
#2.printing 2nd,4th and 7th element of the array
x=np.array([[1,2,3,4,5,6,7]])
print(x[0,1],x[0,3],x[0,6])
#3.printing the shape of the array
X=np.array([[1,2,3],[3,2,1],[5,6,7]])
print(np.shape(X))
#4.printing the transpose of the array
Y=np.array([[1,2,3],[3,2,1]]).T
print(Y)
#5.Creating a 4X5 vector matrix
matrix=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(matrix)
#6. printing the shape of the matrix
matrix=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(np.shape(matrix))
#7. printing the transpose of the matrix
matrix=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]).T
print(matrix)
#8 printing the first column of the matrix
print("First column")
matrix=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(matrix[0:1])
#9 printing the first row of the matrix
print("First row")
matrix=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(matrix[:,0:1])
#10. Creating 7 dimensional array with only 0s. 
print(np.array([[0,0,0,0,0,0,0]]*7))
# Creating 3 dimensional array with only 1s.
print(np.array([[1,1,1]]*3))
