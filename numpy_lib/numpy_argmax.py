import numpy as np

a = np.zeros((4, 3))
a[0, 1] = 1
a[0, 2] = 1
a[0, 0] = 1
a[0, 1] = 1

b = np.zeros((10, 3))
b[1, 1] = 1
b[1, 2] = 1
b[1, 0] = 1
b[1, 1] = 1

print(np.equal(a, b))