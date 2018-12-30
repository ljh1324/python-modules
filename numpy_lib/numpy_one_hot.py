import numpy as np

a = np.array([1, 0, 3])
b = np.zeros((3, 4))
b[np.arange(3), a] = 1
print(b)