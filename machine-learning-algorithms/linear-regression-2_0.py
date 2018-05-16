import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#calculate the Sum of Squared Error
def error(m, b, data):
    sum_of_squares = 0
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        sum_of_squares += (y-(m*x + b))**2
    return sum_of_squares/len(data)

def gradient(m, b, data):
    m_final = 0
    b_final = 0
    N = len(data)
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        b_final += -(2/N)*(y-(m*x+b))
        m_final += -(2/N)*x*(y-(m*x+b))
    return m_final, b_final


def gradient_descent(m, b, iterations, learning_rate, data):
    for i in range(iterations):
        m_final, b_final = gradient(m, b, data)
        m = m - (learning_rate*m_final)
        b = b - (learning_rate*b_final)
    return m, b


def start():
    pdata = pd.read_csv("../datasets/lactic.csv", usecols=['X', 'Y'])
    pdata = np.array(pdata)

    #learning conditions
    learning_rate = 0.001
    iterations = 1000

    #equation mx + b
    m0 = 0            #initial values of coefficients
    b0 = 0

    input_val = []
    output_val = []
    for i in range(len(pdata)):
        input_val.append(pdata[i,0])
        output_val.append(pdata[i,1])


    print("Initial value of coefficients y=mx+b:\n "
          "m = {0} b = {1} and error is {2} \n".format(m0, b0, error(m0, b0, pdata)))

    print("Gradient descent...\n")
    m, b = gradient_descent(m0, b0, iterations, learning_rate, pdata)

    print("Final value of coefficients y=mx+b:\n "
          "m = {0} b = {1} and error is {2}".format(m, b, error(m, b, pdata)))


    #not the best way to represent fit line
    #representation of linear regression results
    y_vals = [m*x+b for x in range(20)]
    x_vals = range(20)

    plt.scatter(input_val, output_val)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.plot(x_vals, y_vals,color='blue', linewidth = 3)
    plt.show()



if __name__ == '__main__':
    start()
