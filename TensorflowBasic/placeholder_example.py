import tensorflow as tf

x = tf.placeholder(tf.float32)

pow_2 = tf.pow(x, 2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    for i in range(10):
        print(sess.run(pow_2, feed_dict={x:i}))