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

wc1 = init_weights([5, 5, 3, 32], 'W1')
bc1 = init_bias([32], 'B1')
wc2 = init_weights([5, 5, 32, 64], 'W2')      # w1에서 pooling을 거친후
bc2 = init_bias([64], 'B2')
wd1 = init_weights([128 * 128 * 64, 1024], 'W3')     # w2에서 pooling을 거치지 않는다
bd1 = init_bias([1024], 'B3')
w_out = init_weights([1024, 2], 'W_OUT')
bout = init_bias([2], 'B_OUT')

param_list = [wc1, wc2, wd1, w_out]
saver = tf.train.Saver(param_list)

#l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))
l1a = conv2d(X, wc1, bc1)
l1b = max_pool(l1a, 2)
conv1 = tf.nn.dropout(l1b, p_keep_conv)
print("l1a: ", end=" ")
print(l1a)
print("l1b: ", end= " ")
print(l1b)
print("l1: ", end= " ")
print(conv1)

l2a = conv2d(conv1, wc2, bc2)
l2b = max_pool(l2a, 2);
conv2 = tf.nn.dropout(l2a, p_keep_conv)
print("l2a: ", end=" ")
print(l2a)
print("l2: ", end= " ")
print(conv2)

# 완전 연결 에이어
# 완전 연결 레이어 형성을 위해 3차원의 conv2를 1차원으로 변환
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
# ReLu 활성화 함수 적용
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
# 드롭아웃 적용
dense1 = tf.nn.dropout(dense1, p_keep_hidden)
# 클래스 예측
pred = tf.add(tf.matmul(dense1, w_out), bout)

# 비용 함수와 최적화 함수 설정
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
#optimizer = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

# 모델 성능 평가
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 변수 초기화
init = tf.global_variables_initializer()

# Load
#print(' < load realfire data >')
#realfire_images = load_image_data('./data_set/real_image.txt')
#realfire_labels = load_result_data('./data_set/real_result.txt')

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

print(len(nonfire_labels))
print(len(nonfire_labels2))
print(len(nonfire_labels3))

print(len(composefire_labels))
print(len(composefire_labels2))
print(len(composefire_labels3))

for i in range(100):
    print(nonfire_labels[i])
    print(nonfire_labels2[i])
    print(nonfire_labels3[i])
    print()

#exit()

print(' < load test_set > ')
test_image_set = load_image_data('./data_set/test_set.txt')
test_result_set = load_result_data('./data_set/test_set_result.txt')

accuracyList = []

# Launch the graph in a session
with tf.Session() as sess:
    sess.run(init)

    print(sess.run([w1, w2, w3, w4, w_out]))

    for i in range(10000):
        print(str(i + 1) + 'session learning!')
        print()

        print('< step1 with nonfire1 data >')
        print()
        sess.run(optimizer, feed_dict={X: nonfire_image, Y:nonfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        #sess.run(optimizer2, feed_dict={X: nonfire_image, Y: nonfire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        #print(sess.run([w1, w2, w3, w4, w_out]))
        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y:test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        print('< step2 with nonfire2 data >')
        print()

        sess.run(optimizer, feed_dict={X: nonfire_image2, Y: nonfire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        #sess.run(optimizer2, feed_dict={X: nonfire_image2, Y: nonfire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        print('< step3 with nonfire3 data >')
        print()
        sess.run(optimizer, feed_dict={X: nonfire_image3, Y: nonfire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        #sess.run(optimizer2, feed_dict={X: nonfire_image3, Y: nonfire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)


        print('< step4 with composefire_fire1 >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images.data, Y:composefire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        #sess.run(optimizer2, feed_dict={X: composefire_images.data, Y: composefire_labels, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)


        print('< step5 with composefire_fire2 >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images2, Y:composefire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        #sess.run(optimizer2, feed_dict={X: composefire_images2, Y: composefire_labels2, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)


        print('< step6 with composefire_fire3 >')
        print()
        sess.run(optimizer, feed_dict={X: composefire_images3.data, Y:composefire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        #sess.run(optimizer2, feed_dict={X: composefire_images3.data, Y: composefire_labels3, p_keep_conv: 0.8, p_keep_hidden: 0.8})

        result = accuracy.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        cost_val = cost.eval(feed_dict={X: test_image_set, Y: test_result_set, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        accuracyList.append(result)
        print(result)
        print(cost_val)

        print(sess.run(cost, feed_dict={X: test_image_set, Y: test_result_set, p_keep_hidden: 1.0, p_keep_conv: 1.0}))

    saver.save(sess, './my-model.ckpt')

for acc in accuracyList:
    print(acc)