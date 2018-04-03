#A =  [[1 2]
#      [3 4]]

#B = [[-2.   1. ]
#     [ 1.5 -0.5]]

#C = [[1 2 3]
#     [4 5 6]]
#Given matrix A and B, mutiply AB - call this mat1. Mutiply BA - call this mat2. Are these matrix inverses?
#Given matrix C, create an identity matrix - call it id1 to multiply C*id1- call this mat3.
#Given matrix C, create an identity matrix - call it id2 to multiply id2*C- call this mat4.

import numpy as np
A = np.matrix(
    [[1, 2],
     [3, 4]]
)

B = np.matrix(
    [[-2, 1],
     [1.5, 0.5]]
)

C = np.matrix(
    [[1, 2, 3],
     [4, 5, 6]]
)
mat1 = A*B
mat2 = B*A
id1 = np.eye(3)
mat3 = C*id1
id2 = np.eye(2)
mat4 = id2*C
