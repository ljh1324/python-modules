import tensorflow as tf
import functions

c1, c2 = tf.constant([3]), tf.constant([1, 5])
v1, v2 = tf.Variable([5]), tf.Variable([2, 4])

functions.showConstant(c1)  # [3]
functions.showConstant(c2)  # [1 5]
functions.showVariable(v1)  # [5]
functions.showVariable(v2)  # [2 4]

print('-----------add------------')
functions.showOperation(tf.add(c1, v1))         # [8]
functions.showOperation(tf.add(c2, v2))         # [3 9]
functions.showOperation(tf.add([c2, v2], [c2, v2]))   # [[ 2 10] [ 4  8]]

print('-----------sub------------')
functions.showOperation(tf.subtract(c1, v1))         # [-2]
functions.showOperation(tf.subtract(c2, v2))         # [-1  1]

print('-----------mul------------')
functions.showOperation(tf.multiply(c1, v1))         # [15]
functions.showOperation(tf.multiply(c2, v2))         # [ 2 20]

print('-----------div------------')             # 정수형 나누기
functions.showOperation(tf.div(c1, v1))         # [0]
functions.showOperation(tf.div(c2, v2))         # [0 1]

print('-----------truediv------------')         # 실수형 나누기
functions.showOperation(tf.truediv(c1, v1))     # [ 0.6]
functions.showOperation(tf.truediv(c2, v2))     # [ 0.5  1.25]

print('-----------floordiv------------')        # floor(바닥) 나누기
functions.showOperation(tf.floordiv(c1, v1))    # [0]
functions.showOperation(tf.floordiv(c2, v2))    # [0 1]

print('-----------mod------------')
functions.showOperation(tf.mod(c1, v1))         # [3]
functions.showOperation(tf.mod(c2, v2))         # [1 1]


# 출처: http://pythonkim.tistory.com/63