import numpy as np
import matplotlib.pyplot as plt

s = np.sin(2 * np.pi * 0.125 * np.arange(20))
plt.plot(s, 'ro-')
plt.xlim(-0.5, 20.5)
plt.ylim(-1.1, 1.1)
plt.show()

from scipy.linalg import toeplitz
# np.r_ : A concatenated ndarray or matrix. https://docs.scipy.org/doc/numpy/reference/generated/numpy.r_.html
S = np.fliplr(toeplitz(np.r_[s[-1], np.zeros(s.shape[0] - 2)], s[::1]))
print(S[:5, :3])

X_train = S[:1, :3][:, :, np.newaxis]
Y_train = S[:-1, 3]
print(X_train.shape, Y_train.shape)
