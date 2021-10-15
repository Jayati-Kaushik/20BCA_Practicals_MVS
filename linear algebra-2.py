#1.Define and print a 6 dimensional vector
import numpy as np
X=np.array([[10,11,12,13,14,15]])
print(X)

#2.Print the transpose of the above vector
print(X.T)

#3.Define two non square matrices such that they can be multiplied.
A=np.array([[1,2],[3,4],[5,6]])
B=np.array([[1,2,3],[4,5,6]])

#4.print the shape of the above matrices
print(np.shape(A),np.shape(B))

#5.Print the product of above two matrices(do so without using the inbuilt functions).
#3X2 matrix
X=[[1,2],
   [3,4],
   [5,6]]

#2X3 matrix
Y=[[1,2,3],
  [4,5,6]]

#result is 3X3
result=[[0,0,0],
        [0,0,0],
        [0,0,0]]

#iterating through rows of X
for i in range(len(X)):
    #iterating through columns of Y
    for j in range(len(Y[0])):
        #iterating through rows of Y
        for k in range(len(Y)):
            result[i][j]=result[i][j]+X[i][k]*Y[k][j]
for r in result:
 print(r)  
 
#6. Define two non square matrices of same order and print their sum.
X=np.array([[1,2,3],[4,5,6]])
Y=np.array([[4,3,2],[7,8,2]])
print(X+Y)

#7.Define a square matrix A
A=np.array([[3,4,8],[5,6,7],[7,6,5]])

#8.Print the transpose of A
print(A.T)

#9.Print the identity matrix of the above order I.
I=np.array([[1,0,0],[0,1,0],[0,0,1]])
print(I)

# 10.Verify A.I=I.A for matrix multiplication.
A=np.array([[3,4,8],[5,6,7],[7,6,5]])
I=np.array([[1,0,0],[0,1,0],[0,0,1]])
X=A@I
print("A.I=",X)
Y=I@A
print("I.A=",Y)
print("Therefore verified, A.I=I.A")

#11. Define another square matrix of the same order as A
B=np.array([[2,1,9],[2,4,8],[1,6,8]])

#12.Print the product of the matrices as matrix multiplication
print(A@B)

#13.Print the product of the matrices by element wise multiplication
print(np.multiply(A,B))

#14.Calculate and print the inverse of A.(Use linalg)
#Check if determinant is 0
#Use if else statement to calculate inverse only when determinant is non 0.
z=np.linalg.det(A)
print("The determinant is=",z)
if z!=0:
    print("The inverse of A is=",np.linalg.inv(A))
else:
    print("Inverse does not exist")


