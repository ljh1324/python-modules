import tensorflow as tf
import numpy as np

xy = np.loadtxt("train3.txt", unpack=True, dtype='float32')

x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3])      # 3: x1, x2 and 1 (for bias)
Y = tf.placeholder("float", [None, 3])      # A, B, C => 3 classes

# Set model weights
W = tf.Variable(tf.zeros([3, 3]))

# Construct model
hypothesis = tf.nn.softmax(tf.matmul(X, W)) # softmax

# Minimize error cross entropy
learning_rate = 0.1

# Cross Entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))      # reduction indices=1 => 행기준으로 평균을 구한다

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(5000):
    sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict = {X: x_data, Y: y_data}), sess.run(W))

test_data = [[1, 0, 0]]
test_data2 = [[1, 20, 20]]

a = sess.run(hypothesis, feed_dict = {X: test_data})
a2 = sess.run(hypothesis, feed_dict = {X: test_data2})

print(a)
print(sess.run(tf.arg_max(a, 1)))
print(a2)
print(sess.run(tf.arg_max(a2, 1)))
