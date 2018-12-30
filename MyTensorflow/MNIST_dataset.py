from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print()
#print(mnist.happy_song.images[0])

idx = 0
for i in range(28):
    for j in range(28):
        if mnist.test.images[0][idx] != 0:
            print("1", end="")
        else:
            print("0", end="")
        idx = idx + 1
    print()

org_label = tf.Variable(mnist.test.labels[0:2])
init = tf.global_variables_initializer()
sess = tf.Session()



#arg_max = tf.argmax(org_label, 1)
#print(sess.run(arg_max))
sess.run(init)
print(sess.run(org_label))
print(mnist.test.labels)
print(mnist.test.labels[0:2])
print(mnist.test.labels)
print(str(mnist.test.num_examples))


# MNIST VARIABLES INITIALIZE !
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print(trX[0])
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img, (Input Row by 28 by 28 by 1)
print(trX[0])








