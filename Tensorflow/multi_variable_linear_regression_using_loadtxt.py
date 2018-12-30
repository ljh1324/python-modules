import tensorflow as tf
import numpy as np

# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.loadtxt.html
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
print(xy)
print()

x_data = xy[0:-1]
y_data = xy[-1]

xy2 = np.loadtxt('train.txt', unpack=False, dtype='float32')
print(xy2)
print()

x_data2 = xy2[0:-1]
y_data2 = xy2[-1]

print('x', x_data)
print('y', y_data)

print('x2', x_data2)
print('y2', y_data2)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -5.0, 5.0))
#W = tf.Variable(tf.random_uniform([len(x_data), 1], -5.0, 5.0))

# Out hypothesis
hypothesis = tf.matmul(W, x_data)

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initaialize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))