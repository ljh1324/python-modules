
import numpy as np

x = np.array([ [ [1, 10],[2, 20],[3, 30] ],
               [ [4, 40],[5, 50],[6, 60] ],
               [ [7, 70],[8, 80],[9, 90] ]
             ])
x2 = np.array( [[1, 2], [2, 3], [4, 5]])
print(x.shape)
print(x2.shape)
print(x[1:2])

print('\nx[...]:', x[...].shape)
print(x[...])
print('\nx[0, ...]:', x[0, ...].shape)
print(x[0, ...])
print('\nx[..., 0]:', x[..., 0].shape)
print(x[..., 0])
print('\nx:', x.shape)
print(x)
print('\nx[..., np.newaxis]:', x[..., np.newaxis].shape)
print(x[..., np.newaxis])
print('\nx[:][:][0]:', x[:][:][0].shape)
print(x[:][:][0])
print('\nx[:]:', x[:].shape)
print(x[:])
print('\nx[:, :, 0]:', x[:, :, 0].shape)
print(x[:, :, 0])
print()
print(x[..., 0].shape)
print(x2[...,0])

test = np.full((1024, 1024, 11), 1)
print(test[..., 0].shape)
print(test[:][:][0].shape)
print(x[0, 0, 1])

