#Find the norm of the vector [3, 9, 5, 4] using the actual formula above. You should write a function find_norm(v1) that returns this value as a float and then call it on the provided variable n1. You should not use scipy, but you may use the math module.

import math 
import numpy as np
A = [3, 9, 5, 4]
def find_norm(v1):
    list_ = [i**2 for i in v1]
    return math.sqrt(sum(list_))

find_norm(A)
