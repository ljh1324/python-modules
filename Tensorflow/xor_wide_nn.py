# Use 'wide' neural network (10 neurals) to solve XOR problem.

import numpy as np
import tensorflow as tf

xy = np.loadtxt('train_xor.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Wide network: Use more neurons in each layer.
W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([10]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Hypotheses
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)                  # x_data:(4행 2열) * W1(2행 10열) = (4행 10열)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)         # L2(4행 10열) * W2(10행 1열) = (4행 1열)

# Cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))  # for logistic regression

# Minimize cost.
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initializa all variables.
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(8001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(step)
            print(sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            print(sess.run(W1))
            print(sess.run(W2))

    # sigmod 함수 값에 0.5를 더한고 floor을 했는 결과값 Y와 같은지 확인한다.
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    # correnct_prediction값을 float형으로 casting 한 후 평균을 낸다
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))
