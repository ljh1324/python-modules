import data_input as di
import tensorflow as tf

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def init_weights(shape, var_name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, name=var_name))

def init_bias(shape, var_num):
    return tf.Variable(tf.zeros(shape), var_num)

def load_image_data(file):
    return di.ImageDataSet(file).data

def load_result_data(file):
    return di.LabelDataSet(file).data

def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Init Values
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CHANNEL = 3
CLASS = 2
p_keep_conv = 0.8
p_keep_hidden = 0.7

# Make Graph
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
Y = tf.placeholder(tf.float32, [None, CLASS])

# AlexNet
w1 = init_weights([11, 11, 3, 96], 'W1')
b1 = init_bias([96], 'B1')
w2 = init_weights([5, 5, 96, 256], 'W2')      # w1에서 pooling을 거친후
b2 = init_bias([256], 'B2')
w3 = init_weights([3, 3, 256, 384], 'W3')     # w2에서 pooling을 거치지 않는다
b3 = init_bias([384], 'B3')
w4 = init_weights([3, 3, 384, 384], 'W4')     # w3에서 pooling을 거친다
b4 = init_bias([384], 'B4')
w5 = init_weights([3, 3, 384, 256], 'W5')     # w4에서 pooling을 거친다
b5 = init_bias([256], 'W5')
w6 = init_weights([16 * 16 * 256, 32768], 'W6')
b6 = init_bias([32768], 'B6')
w_out = init_weights([32768, 2], 'W_OUT')
b_out = init_bias([2], 'B_OUT')

param_list = [w1, w2, w3, w4, w_out]
saver = tf.train.Saver(param_list)

#l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))
l1a = conv2d(X, w1, b1)
l1b = max_pool(l1a, 2)
l1 = tf.nn.dropout(l1b, p_keep_conv)
print("l1a: ", end=" ")
print(l1a)
print("l1b: ", end= " ")
print(l1b)
print("l1: ", end= " ")
print(l1)

l2a = conv2d(l1, w2, b2)
l2 = tf.nn.dropout(l2a, p_keep_conv)
print("l2a: ", end=" ")
print(l2a)
print("l2: ", end= " ")
print(l2)

l3a = conv2d(l2, w3, b3)
l3b = max_pool(l3a, 2)
l3 = tf.nn.dropout(l3b, p_keep_hidden)
print("l3a: ", end=" ")
print(l3a)
print("l3b: ", end= " ")
print(l3b)
print("l3: ", end= " ")
print(l3)

l4a = conv2d(l3, w4, b4)
l4b = max_pool(l4a, 2)
l4 = tf.nn.dropout(l4b, p_keep_conv)
print("l4a: ", end=" ")
print(l4a)
print("l4b: ", end= " ")
print(l4b)
print("l4: ", end= " ")
print(l4)

l5a = conv2d(l4, w5, b5)
l5b = max_pool(l5a, 2)
l5c = tf.reshape(l5b, [-1, w6.get_shape().as_list()[0]])
l5 = tf.nn.dropout(l5c, p_keep_conv)
print("l5a: ", end=" ")
print(l5a)
print("l5b: ", end= " ")
print(l5b)
print("l5c: ", end= " ")
print(l5c)
print("l5: ", end= " ")
print(l5)

l6a = tf.nn.relu(tf.matmul(l5, w6))
l6 = tf.nn.dropout(l6a, p_keep_hidden)
pyx = tf.matmul(l6, w_out)
predict_op = tf.argmax(pyx, 1)         # 예측값의 argmax
print("l6a: ", end= " ")
print(l6a)
print("l6: ", end= " ")
print(l6)
print("pyx: ", end=" ")
print(pyx)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pyx, Y))
#optimizer = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Load
print(' < load realfire data >')
realfire_images = load_image_data('./data_set/real_image.txt')
realfire_labels = load_result_data('./data_set/real_result.txt')

print(' < load nonfire data >')
nonfire_image = load_image_data('./data_set/nonfire_image1.txt')
nonfire_labels = load_result_data('./data_set/nonfire_result1.txt')

nonfire_image2 = load_image_data('./data_set/nonfire_image2.txt')
nonfire_labels2 = load_result_data('./data_set/nonfire_result2.txt')

nonfire_image3 = load_image_data('./data_set/nonfire_image3.txt')
nonfire_labels3 = load_result_data('./data_set/nonfire_result3.txt')

print(' < load composefire data >') # 합성된 fire 사진 불러오기
composefire_labels = load_result_data('./data_set/compose_fire_result1.txt')
composefire_images = load_image_data('./data_set/compose_fire1.txt')

composefire_labels2 = load_result_data('./data_set/compose_fire_result2.txt')
composefire_images2 = load_image_data('./data_set/compose_fire2.txt')

composefire_labels3 = load_result_data('./data_set/compose_fire_result3.txt')
composefire_images3 = load_image_data('./data_set/compose_fire3.txt')

print(' < load test_set > ')
test_image_set = load_image_data('./data_set/test_set.txt')
test_result_set = load_result_data('./data_set/test_set_result.txt')


# Launch the graph in a session
with tf.Session() as sess:
    sess.run(init)

    print(sess.run([w1, w2, w3, w4, w_out]))

    for i in range(10):
        print(str(i + 1) + 'session learning!')
        print()

        print('< step1 with real fire and nonfire1 data >')
        print()
        sess.run(optimizer, feed_dict={X: realfire_images, Y:realfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})      # 왜인지는 모르겠지만 결과값이 [4352, 2]가 나온다
        sess.run(optimizer, feed_dict={X: nonfire_image, Y:nonfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        # happy_song
        #print(pyx.eval(feed_dict={X: test_images, Y:test_labels, p_keep_hidden:1.0, p_keep_conv:1.0}))
        #print(sess.run(Y, feed_dict={Y: test_labels}))

        #print(accuracy.eval(feed_dict={X: test_images, Y:test_labels, p_keep_hidden:1.0, p_keep_conv:1.0}))
        print(sess.run([w1, w2, w3, w4, w_out]))
        print(accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

        print('< step2 with real_fire and nonfire2 data >')
        print()
        sess.run(optimizer, feed_dict={X: realfire_images, Y:realfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        sess.run(optimizer, feed_dict={X: nonfire_image2, Y: nonfire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        #print(accuracy.eval(feed_dict={X: test_images2, Y: test_labels2, p_keep_hidden:1.0, p_keep_conv:1.0}))
        print(sess.run([w1, w2, w3, w4, w_out]))
        print(accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

        print('< step3 with real_fire and nonfire3 data >')
        sess.run(optimizer, feed_dict={X: realfire_images, Y:realfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        sess.run(optimizer, feed_dict={X: nonfire_image3, Y: nonfire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        #print(accuracy.eval(feed_dict={X: test_images3, Y: test_labels3, p_keep_hidden:1.0, p_keep_conv:1.0}))
        print(sess.run([w1, w2, w3, w4, w_out]))
        print(accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

        print('< load compose_fire1 data >')
        print()
        #composefire_labels = di.LabelDataSet('compose_fire_result1.txt')  # 68개 데이터
        #composefire_images = di.ImageDataSet('compose_fire1.txt')  # 68개 데이터


        print('< step4 with composefire_fire1 and nonfire1 data >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images.data, Y:composefire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.5})
        sess.run(optimizer, feed_dict={X: nonfire_image.data, Y: nonfire_labels.data, p_keep_conv: 0.8, p_keep_hidden: 0.5})

        #print(accuracy.eval(feed_dict={X: test_images, Y: test_labels, p_keep_hidden:1.0, p_keep_conv:1.0}))
        print(sess.run([w1, w2, w3, w4, w_out]))
        print(accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

        print('< load compose_fire2 data >')
        print()
        #composefire_labels = di.LabelDataSet('compose_fire_result2.txt')  # 68개 데이터
        #composefire_images = di.ImageDataSet('compose_fire2.txt')  # 68개 데이터
        #composefire_lables = load_result_data('compose_fire_result2.txt')
        #composefire_images = load_image_data('compose_fire2.txt')

        print('< step5 with composefire_fire2 and nonfire2 data >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images2, Y:composefire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.5})
        sess.run(optimizer, feed_dict={X: nonfire_image2, Y: nonfire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.5})

        #print(accuracy.eval(feed_dict={X: test_images2, Y: test_labels2, p_keep_hidden:1.0, p_keep_conv:1.0}))
        print(sess.run([w1, w2, w3, w4, w_out]))
        print(accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

        print('< load compose_fire3 data >')
        print()
        #composefire_labels = di.LabelDataSet('compose_fire_result3.txt')  # 68개 데이터
        #composefire_images = di.ImageDataSet('compose_fire3.txt')  # 68개 데이터
        #composefire_lables = load_result_data('compose_fire_result3.txt')
        #composefire_images = load_image_data('compose_fire3.txt')

        print('< step6 with composefire_fire3 and nonfire3 data >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images3.data, Y:composefire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.5})
        sess.run(optimizer, feed_dict={X: nonfire_image3.data, Y: nonfire_labels3.data, p_keep_conv: 0.8, p_keep_hidden: 0.5})

        #print(accuracy.eval(feed_dict={X: test_images3, Y: test_labels3, p_keep_hidden:1.0, p_keep_conv:1.0}))
        print(sess.run([w1, w2, w3, w4, w_out]))
        print(accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))
        print(sess.run(cost, feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

    saver.save(sess, './my-model.ckpt')