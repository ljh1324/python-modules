import tensorflow as tf

a = tf.Variable([[1., 2.], [3., 4.]])
b = tf.Variable([[1, 2], [3, 4]])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(tf.reduce_mean(a, reduction_indices=1)))