import tensorflow as tf
import matplotlib.pyplot as plt

uniform = tf.random_uniform([100], minval = 0, maxval = 1, dtype=tf.float32)        # 1차원 텐서. 100개의 값이 0부터 1사이의 균일 분포, 즉 동일한 확률로 분포되도록 구성한다.

sess = tf.Session()

with tf.Session() as session:
    print(uniform.eval())
    plt.hist(uniform.eval())
    plt.show()
