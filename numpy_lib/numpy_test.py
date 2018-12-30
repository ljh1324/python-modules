import numpy as np

k = np.full((3, 3), 1)
for i in range(k.shape[0] - 1):
    for j in range(k.shape[1] - 1):
        k[i, j] = 2
print(k)
print(k[:2, :2])
k[:2, :2] = 3
print(k)
print(k[:2, :2])

k = np.full((3, 3), 7)
print(k)

print(np.array([k]))

from skimage.io import imsave, imread
import os
img = imread(os.path.join('test_folder', 'frog.jpg'), as_grey=True)
print(img.shape)
print(img)
img_arr = np.array(img)
img_arr2 = np.array([img])
print(img_arr.shape)
print(img_arr2.shape)
imgs = np.ndarray((10, img.shape[0], img.shape[1]), dtype=np.float32)
print(imgs.shape)
imgs[0] = img
print(img_arr)
print(imgs[0])

list = [3, 3, 3]
print(list[np.nonzero([1, 0, 1])])
