import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_xor.txt', unpack=True)
print(xy)
x_data = xy[0:-1]           # xy[]에서 0 ~ -1행 복사
y_data = xy[-1]

print(x_data)
print(y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))   # for logistic regression

# Minimize
a = tf.Variable(0.1)   # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables. We will 'run' first
init = tf.global_variables_initializer()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    # Fit the line.
    for step in range(3000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data}))
    print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))
