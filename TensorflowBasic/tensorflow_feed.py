import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

input3 = tf.placeholder(tf.float32, [None, 10])

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
  print(sess.run(input3, feed_dict={input3: np.arange(30).reshape(3, 10)}))