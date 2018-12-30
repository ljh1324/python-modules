# Implement code for Sung Kim's TF lecture. See https://www.youtube.com/watch?v=9i7FBbcZPMA&feature=youtu.be

import numpy as np
import tensorflow as tf

xy = np.loadtxt('train_xor.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name = 'X-input')
Y = tf.placeholder(tf.float32, name = 'Y-input')

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name = "Weight1")
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name = "Weight2")

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Out hypothesis
with tf.name_scope("layer2") as scope:  # Grouping
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))
    cost_summ = tf.scalar_summary("cost", cost)

# Minimize cost.
with tf.name_scope("train") as scope:
    a = tf.Variable(0.01)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

w1_hist = tf.histogram_summary("weight1", W1)
w2_hist = tf.histogram_summary("weight2", W2)

b1_hist = tf.histogram_summary("biases1", b1)
b2_hist = tf.histogram_summary("biases2", b2)

y_hist = tf.histogram_summary("y", Y)

# Initializa all variables.
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    merged = tf.merge_all_summaries()   # 모든 summary가 담기게 된다
    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph_def) # 어느 곳에 쓸것인가.

    for step in range(100000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 1000 == 0:
            print(
                step,
                sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                sess.run(W1),
                sess.run(W2)
            )
        if step % 2000 == 0:
            summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})
            writer.add_summary(summary, step)


    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))