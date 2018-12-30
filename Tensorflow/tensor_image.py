import matplotlib.image as mp_image

filename = "D:\\MyPythonProject\\Tensorflow\\frog.png"

input_image = mp_image.imread(filename)

print('input dim = {}'.format(input_image.ndim))
print('input shape = {}'.format(input_image.shape))

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

# import tensorflow as tf
# my_image = tf.placeholder("uint8", [None, None, 3])

# slice = tf.slice(my_image, [10, 0, 0], [16, -1, -1])

# sess = tf.Session()
# result = sess.run(slice, feed_dict = {my_image: input_image})
# print(result.shape)

# plt.imshow(result)
# plt.show()