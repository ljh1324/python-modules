import tensorflow as tf
import data_input as di

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

X = tf.placeholder(tf.float32, [None, 256, 256, 3])
Y = tf.placeholder(tf.float32, [None, 2])
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

w1 = tf.get_variable(shape=[3, 3, 3, 32], initializer=xavier_init(256, 128), name='weight1')
w2 = tf.get_variable(shape=[3, 3, 32, 64], initializer=xavier_init(128, 64), name='weight2')
w3 = tf.get_variable(shape=[3, 3, 64, 16], initializer=xavier_init(64, 32), name='weight3')
w4 = tf.Variable(tf.random_normal([16*32*32, 625], stddev=0.1), name='weight4')
#w_out = init_weights([625, 2])         # for output weight
w_out = tf.Variable(tf.random_normal([625, 2], stddev=0.01), name='weight_out')

param_list = [w1, w2, w3, w4, w_out]
saver = tf.train.Saver(param_list)

l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))
l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(l1, p_keep_conv)
print(l1a)

l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, p_keep_conv)
print(l2a)

l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
print(l3a)
l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(l3)
l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
print(l3)
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, p_keep_hidden)
pyx = tf.matmul(l4, w_out)
predict_op = tf.argmax(pyx, 1)         # 예측값의 argmax

print(l4)
print(pyx)

correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
param_list = [w1, w2, w3, w4, w_out]
saver = tf.train.Saver(param_list)

test_image_set = di.ImageDataSet('test_set.txt').data
test_result_set = di.LabelDataSet('test_set_result.txt').data

with tf.Session() as sess:
    saver.restore(sess, './my-model.ckpt')
    print(sess.run(accuracy, feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv:1.0, p_keep_hidden:1.0}))
