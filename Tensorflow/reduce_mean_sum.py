import tensorflow as tf


x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_data2 = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
x = tf.placeholder(tf.int32, [None])

sum = tf.reduce_sum(x)
mean = tf.reduce_mean(x)

#init = tf.global_variables_initializer()

with tf.Session() as sess:
    print(sess.run(sum, feed_dict={x: x_data}))
    print(sess.run(mean, feed_dict={x: x_data}))
    print(sess.run(sum, feed_dict={x: x_data2}))
    print(sess.run(mean, feed_dict={x: x_data2}))



