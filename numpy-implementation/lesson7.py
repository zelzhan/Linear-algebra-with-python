'''Write a function matrix_inverse(matrix_A) that outputs the inverse matrix.'''

import math
import numpy as np

def matrix_inverse(matrix_A):
    inv = np.linalg.solve(matrix_A, np.eye(matrix_A.shape[0]))
    return inv

C = np.matrix([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
print(matrix_inverse(C))
print(np.linalg.inv(C))
