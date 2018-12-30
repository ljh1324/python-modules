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
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
CHANNEL = 9
CLASS = 2
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

# Make Graph
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
Y = tf.placeholder(tf.float32, [None, CLASS])

# AlexNet

"""
w1 = init_weights([11, 11, 3, 8], 'W1')
b1 = init_bias([8], 'B1')
w2 = init_weights([5, 5, 8, 16], 'W2')      # w1에서 pooling을 거친후
b2 = init_bias([16], 'B2')
w3 = init_weights([3, 3, 16, 32], 'W3')     # w2에서 pooling을 거치지 않는다
b3 = init_bias([32], 'B3')
w4 = init_weights([3, 3, 32, 32], 'W4')     # w3에서 pooling을 거친다
b4 = init_bias([32], 'B4')
w5 = init_weights([3, 3, 32, 16], 'W5')     # w4에서 pooling을 거친다
b5 = init_bias([16], 'W5')
w6 = init_weights([16 * 16 * 16, 1024], 'W6')
b6 = init_bias([1024], 'B6')
w_out = init_weights([1024, 2], 'W_OUT')
b_out = init_bias([2], 'B_OUT')
"""

"""
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
w6 = init_weights([16 * 16 * 256, 2048], 'W6')
b6 = init_bias([2048], 'B6')
w_out = init_weights([2048, 2], 'W_OUT')
b_out = init_bias([2], 'B_OUT')
"""

w1 = init_weights([11, 11, 9, 16], 'W1')
b1 = init_bias([16], 'B1')
w2 = init_weights([5, 5, 16, 32], 'W2')      # w1에서 pooling을 거친후
b2 = init_bias([32], 'B2')
w3 = init_weights([3, 3, 32, 32], 'W3')     # w2에서 pooling을 거치지 않는다
b3 = init_bias([32], 'B3')
w4 = init_weights([3, 3, 32, 32], 'W4')     # w3에서 pooling을 거친다
b4 = init_bias([32], 'B4')
w5 = init_weights([3, 3, 32, 32], 'W5')     # w4에서 pooling을 거친다
b5 = init_bias([32], 'W5')
w6 = init_weights([2 * 2 * 32, 512], 'W6')
b6 = init_bias([512], 'B6')
w_out = init_weights([512, 2], 'W_OUT')
b_out = init_bias([2], 'B_OUT')
# InvalidArgumentError (see above for traceback): Input to reshape is a tensor with 12800 values, but the requested shape requires a multiple of 8192
#	 [[Node: layer5/Reshape = Reshape[T=DT_FLOAT, Tshape=DT_INT32, _device="/job:localhost/replica:0/task:0/cpu:0"](layer5/MaxPool, layer5/Reshape/shape)]]
param_list = [w1, w2, w3, w4, w5, w6, w_out, b1, b2, b3, b4, b5, b6, b_out]
weight_list = [w1, w2, w3, w4, w5, w6, w_out]
biased_list = [b1, b2, b3, b4, b5, b6, b_out]

saver = tf.train.Saver(param_list)

#l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))'

with tf.name_scope("layer1") as scope:  # Grouping
    l1a = conv2d(X, w1, b1)
    l1b = max_pool(l1a, 2)
    l1 = tf.nn.dropout(l1b, p_keep_conv)
print("l1a: ", end=" ")
print(l1a)
print("l1b: ", end= " ")
print(l1b)
print("l1: ", end= " ")
print(l1)

with tf.name_scope("layer2") as scope:  # Grouping
    l2a = conv2d(l1, w2, b2)
    l2 = tf.nn.dropout(l2a, p_keep_conv)
print("l2a: ", end=" ")
print(l2a)
print("l2: ", end= " ")
print(l2)

with tf.name_scope("layer3") as scope:  # Grouping
    l3a = conv2d(l2, w3, b3)
    l3b = max_pool(l3a, 2)
    l3 = tf.nn.dropout(l3b, p_keep_hidden)
print("l3a: ", end=" ")
print(l3a)
print("l3b: ", end= " ")
print(l3b)
print("l3: ", end= " ")
print(l3)

with tf.name_scope("layer4") as scope:  # Grouping
    l4a = conv2d(l3, w4, b4)
    l4b = max_pool(l4a, 2)
    l4 = tf.nn.dropout(l4b, p_keep_conv)
print("l4a: ", end=" ")
print(l4a)
print("l4b: ", end= " ")
print(l4b)
print("l4: ", end= " ")
print(l4)

with tf.name_scope("layer5") as scope:  # Grouping
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

with tf.name_scope("layer6") as scope:  # Grouping
    l6a = tf.nn.relu(tf.matmul(l5, w6))
    l6 = tf.nn.dropout(l6a, p_keep_hidden)
print("l6a: ", end= " ")
print(l6a)
print("l6: ", end= " ")
print(l6)

pyx = tf.matmul(l6, w_out)
print("pyx: ", end=" ")
print(pyx)


with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pyx, labels=Y))
    cost_sum = tf.summary.scalar("cost", cost)

#optimizer = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.000).minimize(cost)

with tf.name_scope('train') as scope:
     optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

predict_op = tf.argmax(pyx, 1)         # 예측값의 argmax
init = tf.global_variables_initializer()
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy_result = tf.summary.scalar("accuracy", accuracy)


w1_hist = tf.summary.histogram("weight1", w1)
w2_hist = tf.summary.histogram("weight2", w2)
w3_hist = tf.summary.histogram("weight3", w3)
w4_hist = tf.summary.histogram("weight4", w4)
w5_hist = tf.summary.histogram("weight5", w5)
w6_hist = tf.summary.histogram("weight6", w6)
w_out_hist = tf.summary.histogram("weight6", w_out)

b1_hist = tf.summary.histogram("biases1", b1)
b2_hist = tf.summary.histogram("biases2", b2)
b3_hist = tf.summary.histogram("biases2", b3)
b4_hist = tf.summary.histogram("biases2", b4)
b5_hist = tf.summary.histogram("biases2", b5)
b6_hist = tf.summary.histogram("biases2", b6)
b_out_hist = tf.summary.histogram("biases2", b_out)

y_hist = tf.summary.histogram("y", pyx)



# InternalError (see above for traceback): Dst tensor is not initialized.           # Memory Allocation Problem Error



# Load
print(' < load nonfire data >')
nonfire_image = load_image_data('./hog/view_image1_hog.txt')
nonfire_labels = load_result_data('./hog/view_result1.txt')

nonfire_image2 = load_image_data('./hog/view_image2_hog.txt')
nonfire_labels2 = load_result_data('./reddish/view_result2.txt')

nonfire_image3 = load_image_data('./hog/view_image3_hog.txt')
nonfire_labels3 = load_result_data('./reddish/view_result2.txt')

print(' < load composefire data >') # 합성된 fire 사진 불러오기
composefire_labels = load_result_data('./reddish/compose_result1.txt')
composefire_images = load_image_data('./hog/compose_image1_hog.txt')

composefire_labels2 = load_result_data('./reddish/compose_result2.txt')
composefire_images2 = load_image_data('./hog/compose_image2_hog.txt')

composefire_labels3 = load_result_data('./reddish/compose_result3.txt')
composefire_images3 = load_image_data('./hog/compose_image3_hog.txt')

print(' < data shake >')
# data mix
for i in range(0, 100, 2):
    image = nonfire_image[i]
    nonfire_image[i] = composefire_images[i]
    composefire_images[i] = image
    label = nonfire_labels[i]
    nonfire_labels[i] = composefire_labels[i]
    composefire_labels[i] = label

    image = nonfire_image2[i]
    nonfire_image2[i] = composefire_images2[i]
    composefire_images2[i] = image
    label = nonfire_labels2[i]
    nonfire_labels2[i] = composefire_labels2[i]
    composefire_labels2[i] = label

    image = nonfire_image3[i]
    nonfire_image3[i] = composefire_images3[i]
    composefire_images3[i] = image
    label = nonfire_labels3[i]
    nonfire_labels3[i] = composefire_labels3[i]
    composefire_labels3[i] = label

#exit()

print(' < load test_set > ')
test_image_set = load_image_data('./hog/test_image_hog.txt')
test_result_set = load_result_data('./reddish/test_result.txt')

accuracyList = []

f = open('D:/MyPythonProject/fire_detector_data_reddish_result.txt', 'w')
f2 = open('D:/MyPythonProject/fire_detector_data_reddish_weight.txt', 'w')
f3 = open('D:/MyPythonProject/fire_detector_data_reddish_biased.txt', 'w')

# Launch the graph in a session
with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()   # 모든 summary가 담기게 된다
    writer = tf.summary.FileWriter("./logs/fire_detector", sess.graph) # 어느 곳에 쓸것인가.

    #print(sess.run([w1, w2, w3, w4, w_out]))

    for i in range(100000):


        print(str(i + 1) + 'session learning!')
        print()

        print('< step1 with nonfire1 data >')
        print()
        sess.run(optimizer, feed_dict={X: nonfire_image, Y:nonfire_labels, p_keep_conv: 0.7, p_keep_hidden: 0.7})
        #sess.run(optimizer2, feed_dict={X: nonfire_image, Y: nonfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        #print(sess.run([w1, w2, w3, w4, w_out]))
        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)

        print(sess.run(predict_op, feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv:1.0, p_keep_hidden:1.0}))
        print(sess.run(tf.argmax(Y, 1), feed_dict={Y: test_result_set}))

        print(sess.run(correct_prediction,
                       feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        print(result)
        print(cost_val)

        print('< step2 with nonfire2 data >')
        print()

        sess.run(optimizer, feed_dict={X: nonfire_image2, Y: nonfire_labels2, p_keep_conv: 0.7, p_keep_hidden: 0.7})
        #sess.run(optimizer2, feed_dict={X: nonfire_image2, Y: nonfire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        print('< step4 with composefire_fire1 >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images.data, Y:composefire_labels, p_keep_conv: 0.7, p_keep_hidden: 0.7})
        #sess.run(optimizer2, feed_dict={X: composefire_images.data, Y: composefire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        print('< step5 with composefire_fire2 >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images2, Y:composefire_labels2, p_keep_conv: 0.7, p_keep_hidden: 0.7})
        #sess.run(optimizer2, feed_dict={X: composefire_images2, Y: composefire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        print('< step6 with composefire_fire3 >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images3.data, Y:composefire_labels3, p_keep_conv: 0.7, p_keep_hidden: 0.7})
        #sess.run(optimizer2, feed_dict={X: composefire_images3.data, Y: composefire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        f.write(str(i + 1) + '\n')
        f2.write(str(i + 1) + '\n')
        f3.write(str(i + 1) + '\n')

        f.write("-predict_op\n")
        f.write(str(sess.run(predict_op, feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv:1.0, p_keep_hidden:1.0})))

        f.write("\n-Y\n")
        f.write(str(sess.run(tf.argmax(Y, 1), feed_dict={Y: test_result_set})))

        f.write("\n-correct_prediction")
        f.write(str(sess.run(correct_prediction,
                       feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
        f.write("\n-accuracy: ")
        f.write(str(result))
        f.write("\n-cost_val: ")
        f.write(str(cost_val))
        f.write('\n\n')

        f2.write(str(sess.run(weight_list)))
        f2.write("-\npredict_op\n")
        f2.write(str(sess.run(predict_op, feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv:1.0, p_keep_hidden:1.0})))

        f2.write("\n-Y\n")
        f2.write(str(sess.run(tf.argmax(Y, 1), feed_dict={Y: test_result_set})))

        f2.write("\n-correct_prediction")
        f2.write(str(sess.run(correct_prediction,
                              feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
        f2.write("\n-accuracy: ")
        f2.write(str(result))
        f2.write("\n-cost_val: ")
        f2.write(str(cost_val))
        f2.write('\n\n')

        f3.write(str(sess.run(biased_list)))
        f3.write("-\npredict_op\n")
        f3.write(str(sess.run(predict_op, feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv:1.0, p_keep_hidden:1.0})))

        f3.write("\n-Y\n")
        f3.write(str(sess.run(tf.argmax(Y, 1), feed_dict={Y: test_result_set})))

        f3.write("\n-correct_prediction")
        f3.write(str(sess.run(correct_prediction,
                              feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
        f3.write("\n-accuracy: ")
        f3.write(str(result))
        f3.write("\n-cost_val: ")
        f3.write(str(cost_val))
        f3.write('\n\n')


    saver.save(sess, './my-model-reddish.ckpt')

f.close()
f2.close()
f3.close()
