'''Write a function matrix_add(matrix_A, matrix_B) that performs matrix addition if the dimensionality is valid. Note that the dimensionality is only valid if input matrix A and input matrix B are of the same dimension in both their row and column lengths.

For example, you can add a 3x5 matrix with a 3x5 matrix, but you cannot add a 3x5 matrix with a 3x1 matrix. If the dimensionality is not valid, print this error message "Cannot perform matrix addition between a-by-b matrix and c-by-d matrix", where you substitute a, b with the dimension of the input matrix A, and c,d with the dimension of the input matrix B.'''

import math
import numpy as np

def matrix_add(matrix_A, matrix_B):
    empty = np.empty([matrix_A.shape[0], matrix_A.shape[1]])
    if matrix_A.shape != matrix_B.shape:
        print("Cannot perform matrix addition between {0}-by-{1} matrix and {2}-by-{3} matrix".format(matrix_A.shape[0], matrix_A.shape[1], matrix_B[0], matrix_B[1]))                                
        return empty
    for i in range(matrix_A.shape[0]):
        for j in range(matrix_B.shape[1]):
            empty[i][j] = matrix_A.tolist()[i][j] + matrix_B.tolist()[i][j]
    return empty

print(matrix_add(A, C))

