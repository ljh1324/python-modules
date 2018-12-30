import tensorflow as tf
import functions

pi = 3.14159265
a = tf.Variable([pi/6., pi/4., pi/3.])
functions.showOperation(tf.sin(a))