#Sharwin 20BCAB03

#Program 1
#create and print 7 tuple 1 D array
import numpy as np
array1=np.array([1,2,3,4,5,6,7])
print(array1)

#Program 2
#print 2nd 4th and 7th element of an array
array2=np.array([1,2,3,4,5,6,7,8,9,12])
print(array2[1])
print(array2[3])
print(array2[6])

#Program 3
#print the shape of an array
array3=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(array3.shape)

#Program 4
#print transpose of your array
array4=np.array([[1,2,3,4],[5,6,7,8]]).T
print(array4)

#Program 5
#create a 4*5 vector matrix
array5=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(array5)

#Program 6
#print the shape of the above matrix
array6=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print(np.shape(array6))

#Program 7
#print the transpose of the matrix
array7=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]).T
print(array7)

#Program 8 & 9
#print the first column of the matrix
#print the first row of the matrix
array8=np.array([[1,2,3,4,5],[5,6,7,8,9],[3,5,6,7,8],[1,4,5,6,7]])
print("First column")
print(array8[:,0])
print("First row")
print(array8[0])

#Program 10
#create 7 dimensional array with only 0s, create a 3 dimensional array with only 1s
print(np.zeros(7))
print(np.ones(3))