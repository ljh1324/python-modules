import tensorflow as tf
import numpy as np

class JTensor:
    def __init__(self):
        pass

    def gradientDescent(self, x_data, y_data, weight, bias, alpha):
        hypothesis = tf.matmul(weight, x_data) + bias;
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        init = tf.global_variables_initializer()
        train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
        with tf.Session() as sess:
            sess.run(init)
            for i in range(10000):
                sess.run(train)

jtensor = JTensor()

xy = np.loadtxt('train.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

weight = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))
bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

jtensor.gradientDescent(x_data, y_data, weight, bias, 0.1)