#Write the representation of ℜ2 as a list comprehension - use ranges between -10 and 10 for all values of x and y.
#Write the representation of ℜ3 as a list comprehension - use ranges between -10 and 10 for all values of x, y, and z.
#Write a list comprehension that represents the the set V = {(x, y, z) | x, y, z ∈ ℜ and x+y = 11}. Use ranges between -10 and 10 for all values of x, y, and z.
#Choose three values of x, y, and z that show the set V = {(x, y, z) | x, y, z ∈ ℜ and x+y = 11} is not a subspace of ℜ3. These values should represent a tuple that would be in vector V had it been a vector subspace. Each value should also be between -10 and 10.

import numpy as np
R2 = [[x, y] for x in range(-10, 10) for y in range(-10, 10)]
R3 = [[x,y,z] for x in range(-10, 10) for y in range(-10,10) for z in range(-10, 10)]
V = [[x,y,z] for x in range(-10, 10) for y in range(-10,10) for z in range(-10, 10) if x + y == 11]                    
not_sub = [[x,y,z] for x in range(-10, 10) for y in range(-10,10) for z in range(-10, 10) if x + y != 11]                    


