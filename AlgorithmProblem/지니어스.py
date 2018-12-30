import numpy as np


def getProb():
    matrix = np.zeros((4 * n, 4 * n))
    for i in range(3 * n):
        matrix[i][i + n] = 1.0

    for i in range(n):
        for j in range(n):
            matrix[3 * n + i][(4 - length[j]) * n + j] = T[j][i]
    print(matrix)
    print()
    for i in range(k):
        matrix = np.dot(matrix, matrix)
        print(matrix)
        print()

if __name__ == '__main__':
    T = [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4], [0.3, 0.3, 0.3]]
    n = 3
    k = 6
    length = [2, 3, 4]
    getProb()
