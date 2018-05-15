'''Write a function matrix_det(matrix_A) that calculates the determinant (an integer) of matrix A if it is a 1x1 or 2x2 matrix.
If the input matrix is not square, print this error message "Cannot calculate determinant of a non-square matrix."
If the input matrix is square but has a higher dimension, print the error message "Cannot calculate determinant of a n-by-n matrix", where you substitute n with the dimension of the input matrix.'''

import math
import numpy as np

def matrix_det(matrix_A):
    if matrix_A.shape[0] > 2  or matrix_A.shape[1] > 2:
        print( "Cannot calculate determinant of a {0}-by-{1} matrix".format(A.shape[0], A.shape[1]))                         
        return     
    if matrix_A.shape[0] != matrix_A.shape[1]:
        print("Cannot calculate determinant of a non-square matrix.")
        return
    x = matrix_A.shape[0]
    y = matrix_A.shape[1]
    if x == 1: return matrix_A[0][0]
    return matrix_A.item(0, 0)*matrix_A.item(1, 1) - matrix_A.item(1, 0)*matrix_A.item(0, 1)

A = np.matrix([[1, 2],
               [3, 4]])
B= np.matrix([[1]])
print(matrix_det(A))
print(np.linalg.det(A))
