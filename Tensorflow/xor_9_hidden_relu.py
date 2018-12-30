import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_xor.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], [4, 1])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0))
W10 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([5]))
b2 = tf.Variable(tf.zeros([5]))
b3 = tf.Variable(tf.zeros([5]))
b4 = tf.Variable(tf.zeros([5]))
b5 = tf.Variable(tf.zeros([5]))
b6 = tf.Variable(tf.zeros([5]))
b7 = tf.Variable(tf.zeros([5]))
b8 = tf.Variable(tf.zeros([5]))
b9 = tf.Variable(tf.zeros([5]))
b10 = tf.Variable(tf.zeros([1]))

L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)
hypothesis = tf.sigmoid(tf.matmul(L9, W10) + b10)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))

alpha = tf.Variable(0.01)
train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(20000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(
                step,
                sess.run(cost, feed_dict={X: x_data, Y: y_data}),
            )
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))
