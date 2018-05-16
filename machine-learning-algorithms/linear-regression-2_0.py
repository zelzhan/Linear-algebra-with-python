import numpy as np
import pandas as pd

#calculate the Sum of Squared Error
def error(m, b, data):
    sum_of_squares = 0
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        sum_of_squares += (y-(m*x + b))**2
    return sum_of_squares/len(data)

def gradient_descent(m, x, iterations, learning_rate):


def start():
    pdata = pd.read_csv("../datasets/lactic.csv", usecols=['X', 'Y'])
    pdata = np.array(pdata)

    #equation mx + b
    m0 = 0            #initial values of coefficients
    b0 = 0

    print("Initial value of coefficients y=mx+b:\n "
          "m = {0} b = {1} and error is {2} \n".format(m0, b0, error(m0, b0, pdata)))

    print("Gradient descent...\n")

    print("Final value of coefficients y=mx+b:\n "
          "m = {0} b = {1} and error is {2}".format(m0, b0, 0))

if __name__ == '__main__':
    start()
