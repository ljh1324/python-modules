import random

#import input_data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)                                           # result matrix = [None, 10]
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), reduction_indices=1))        # cross entropy

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

randomValue = tf.placeholder(tf.float32, [1, 10])

training_epoch = 25
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as session:
    session.run(init)

    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)                        # num of batch
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            session.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            # Compute average loss
            avg_cost += session.run(cost, feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost /= total_batch

        # show logs per epoch step

        if epoch % display_step == 0:  # Softmax
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print (session.run(b))

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    arr = correct_prediction.eval({X: mnist.test.images, Y: mnist.test.labels})

    display = 0
    for flag in arr:
        print(flag, end=' ')
        display = (display + 1) % 30
        if display == 0:
            print()

    print()
    print("Correct_prediction: ", correct_prediction.eval({X: mnist.test.images, Y:mnist.test.labels}))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)

    idx = 0
    for i in range(28):
        for j in range(28):
            if mnist.test.images[r][idx] != 0:
                print("1", end="")
            else:
                print("0", end="")
            idx = idx + 1
        print()

    print("Label: ", session.run(tf.argmax(randomValue, 1), feed_dict={randomValue: mnist.test.labels[r: r + 1]}))
    # 1행 10열 배열을 happy_song.label에서 불러온후 argmax function에 인자로 준다
    print("Prediction: ", session.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))






