import tensorflow as tf

x = tf.Variable([1, 0, 0])   # 행 데이터가 아니라 열 데이터라 볼 수 있다!
x2 = tf.Variable([ [0, 1, 0] ])  # 1 행 3열의 데이터
x3 = tf.Variable([ [0, 0, 1], [1, 0, 0]])

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(TensorflowBasic.argmax(x, 0)))
print()

print(sess.run(TensorflowBasic.argmax(x2, 1)))    # 행의 데이터 중 가장 큰 값의 index 반환
print()

print(sess.run(TensorflowBasic.argmax(x3, 1)))    # 각 행의 데이터 중 가장 큰 값의 index 반환
print()
