import matplotlib.pyplot as plt
import numpy as np

def plot(points, shape, limit):
    for i in range(len(points)):
        plt.plot(points[i].real, points[i].imag, shape)
    #limit = np.max(np.ceil(np.absolute(points)))  # set limits for axis
    plt.xlim((-limit, limit))
    plt.ylim((-limit, limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()

def plot_with_line(points, shape, limit):
    for i in range(len(points)):
        plt.plot([0, points[i].real], [0, points[i].imag], shape)
    #limit = np.max(np.ceil(np.absolute(points)))  # set limits for axis
    plt.xlim((-limit, limit))
    plt.ylim((-limit, limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()
