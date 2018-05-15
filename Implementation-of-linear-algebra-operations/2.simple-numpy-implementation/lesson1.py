#Using the distance formula and trigonometry functions in Python, calculate the #magnitude and direction of a line with the two coordinates, (5,3) and (1,1).

import numpy as np
import math
a = np.array([5, 3])
b = np.array([1, 1])
length = np.linalg.norm(a-b)
angle = math.atan((a-b)[1]/(a-b)[0])

