#Write code for two vectors with five values of your choice. The first should be written as a regular one-dimensional list. The other should be be written with numpy.
import scipy.linalg as nln
import numpy as np

a = [1, 20, 22]
nln.norm(a)
b = np.linalg.norm(np.array(a))

