import data_input as di
import tensorflow as tf

def xavier_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def init_weights(shape) :
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#def load_image_data(file):
#    return tf.cast(di.ImageDataSet(file).data, tf.float32)

def load_image_data(file):
    return di.ImageDataSet(file).data

#def load_result_data(file):
#    return tf.cast(di.LabelDataSet(file).data, tf.float32)

def load_result_data(file):
    return di.LabelDataSet(file).data

# load files
#print(" < load nonfire datas >")
#nonfire_image = di.ImageDataSet('nonfire_image1.txt')  # 100개 데이터
#nonfire_labels = di.LabelDataSet('nonfire_result1.txt')

#nonfire_image_data = tf.cast(nonfire_image.data, "float")
#nonfire_labels_data = tf.cast(nonfire_labels.data, "float")


#nonfire_image2 = di.ImageDataSet('nonfire_image2.txt')
#nonfire_labels2 = di.LabelDataSet('nonfire_result2.txt')

#nonfire_image2_data = tf.cast(nonfire_image2.data, "float")
#nonfire_labels2_data = tf.cast(nonfire_labels2.data, "float")



#nonfire_image3 = di.ImageDataSet('nonfire_image3.txt')
#nonfire_labels3 = di.LabelDataSet('nonfire_result3.txt')

#nonfire_image3_data = tf.cast(nonfire_image3.data, "float")
#nonfire_labels3_data = tf.cast(nonfire_labels3.data, "float")
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

#print(" < load happy_song data set > ")
#test_labels = di.LabelDataSet('compose_fire_result4.txt')  # 100개 데이터
#test_images = di.ImageDataSet('compose_fire4.txt')  # 100개 데이터
#test_image_data = tf.cast(test_images.data, "float")
#test_labels_data = tf.cast(test_labels.data, "float")

#test_labels2 = di.LabelDataSet('compose_fire_result5.txt')  # 100개 데이터
#test_images2 = di.ImageDataSet('compose_fire5.txt')  # 100개 데이터
#test_image_data2 = tf.cast(test_images2.data, "float")
#test_labels_data2= tf.cast(test_labels2.data, "float")

#test_labels3 = di.LabelDataSet('compose_fire_result6.txt')  # 100개 데이터
#test_images3 = di.ImageDataSet('compose_fire6.txt')  # 100개 데이터
#test_image_data3 = tf.cast(test_images3.data, "float")
#test_labels_data3 = tf.cast(test_labels3.data, "float")

print(' < load composefire data >') # 합성된 fire 사진 불러오기
composefire_labels = load_result_data('./data_set/compose_fire_result1.txt')
composefire_images = load_image_data('./data_set/compose_fire1.txt')

composefire_labels2 = load_result_data('./data_set/compose_fire_result2.txt')
composefire_images2 = load_image_data('./data_set/compose_fire2.txt')

composefire_labels3 = load_result_data('./data_set/compose_fire_result3.txt')
composefire_images3 = load_image_data('./data_set/compose_fire3.txt')

#test_labels = load_result_data('compose_fire_result4.txt')
#test_images = load_image_data('compose_fire4.txt')

#test_labels2 = load_result_data('compose_fire_result5.txt')
#test_images2 = load_image_data('compose_fire5.txt')

#test_labels3 = load_result_data('compose_fire_result6.txt')
#test_images3 = load_image_data('compose_fire6.txt')

print(' < load test_set > ')
test_image_set = load_image_data('./data_set/test_set.txt')
test_result_set = load_result_data('./data_set/test_set_result.txt')

#w1 = tf.get_variable("W1", shape=[28, 28, 3, 32], initializer=xavier_init(256, 256))
#w2 = tf.get_variable("W2", shape=[28, 28, 32, 64], initializer=xavier_init(128, 128))
#w3 = tf.get_variable("W3", shape=[28, 28, 64, 128], initializer=xavier_init(128, 128))
#w4 = tf.Variable(tf.random_normal([128*64*64, 32768], stddev=0.1))                         // ValuedError 발생!. Too many elements provided.

#w1 = init_weights([28, 28, 3, 32])
#w2 = init_weights([28, 28, 32, 64])
#w3 = init_weights([28, 28, 64, 128])
#w4 = init_weights([128*64*64, 32768])      // 노드가 많아져서 ResourceExhaustedError 발생!
#w_out = init_weights([32768, 2])

#w1 = init_weights([3, 3, 3, 32])        # 3 by 3 by 1 filter * 32
#w2 = init_weights([3, 3, 32, 64])       # 3 by 3 by 32 filter * 64
#w3 = init_weights([3, 3, 64, 16])      # 3 by 3 by 64 filter * 128
#w4 = init_weights([16*32*32, 625])       # for fully connected weight
#w_out = init_weights([625, 2])         # for output weight



# Make Graph
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

print(l4)
print(pyx)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pyx, Y))
#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(3.0, 0.9).minimize(cost)

init = tf.global_variables_initializer()
predict_op = tf.argmax(pyx, 1)         # 예측값의 argmax

# correct_prediction = tf.equal(tf.argmax(predict_op, 1), tf.argmax(Y, 1))    # InvalidArgumentError (see above for traceback): Minimum tensor rank: 2 but got: 1
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Launch the graph in a session
with tf.Session() as sess:
    sess.run(init)

    #print('< load real_fire >')
    #print()
    #realfire_labels = di.LabelDataSet('real_result.txt')  # 68개 데이터
    #realfire_images = di.ImageDataSet('real_image.txt')  # 68개 데이터
    #realfire_image_data = tf.cast(realfire_images.data, "float")
    #realfire_labels_data = tf.cast(realfire_labels.data, "float")

    #realfire_images = load_image_data('real_image.txt')
    #realfire_labels = load_result_data('real_result.txt')
    #realfire_images = realfire_images.eval()
    #realfire_labels = realfire_labels.eval()

    #nonfire_image = nonfire_image.eval()
    #nonfire_labels = nonfire_labels.eval()

    #nonfire_image2 = nonfire_image2.eval()
    #nonfire_labels2 = nonfire_labels2.eval()

    #nonfire_image3 = nonfire_image3.eval()
    #nonfire_labels3 = nonfire_labels3.eval()

    #composefire_labels = composefire_labels.eval()
    #composefire_images = composefire_images.eval()

    #composefire_labels2 = composefire_labels2.eval()
    #composefire_images2 = composefire_images2.eval()

    #composefire_labels3 = composefire_labels3.eval()
    #composefire_images3 = composefire_images3.eval()

    #test_labels = test_labels.eval()
    #test_images = test_images.eval()

    #test_labels2 = test_labels2.eval()
    #test_images2 = test_images2.eval()

    #test_labels3 = test_labels3.eval()
    #test_images3 = test_images3.eval()

    #test_result_set = test_result_set.eval()
    #test_image_set = test_image_set.eval()

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
    # end for statement

    #realfire_labels = di.LabelDataSet('compose_fire_result4.txt')  # 100개 데이터
    #realfire_images = di.ImageDataSet('compose_fire4.txt')  # 100개 데이터
    #realfire_labels = di.LabelDataSet('compose_fire_result5.txt')  # 100개 데이터
    #realfire_images = di.ImageDataSet('compose_fire5.txt')  # 100개 데이터
    #realfire_labels = di.LabelDataSet('compose_fire_result6.txt')  # 100개 데이터
    #realfire_images = di.ImageDataSet('compose_fire6.txt')  # 100개 데이터
    #realfire_labels = di.LabelDataSet('compose_fire_result7.txt')  # 100개 데이터
    #realfire_images = di.ImageDataSet('compose_fire7.txt')  # 100개 데이터
    #realfire_labels = di.LabelDataSet('compose_fire_result8.txt')  # 100개 데이터
    #realfire_images = di.ImageDataSet('compose_fire8.txt')  # 100개 데이터
    #realfire_labels = di.LabelDataSet('compose_fire_result9.txt')  # 24개 데이터
    #realfire_images = di.ImageDataSet('compose_fire9.txt')  # 24개 데이터
