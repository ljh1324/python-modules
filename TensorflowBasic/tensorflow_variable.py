import tensorflow as tf
# 변수를 하나 생성하고 스칼라 값인 0으로 초기화합니다.
state = tf.Variable(0, name="counter")

# one을 state에 더하는 연산을 생성합니다.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 변수는 그래프가 올라간 뒤 'init' 연산을 실행해서 반드시 초기화 되어야 합니다.
# 그 전에 먼저 'init' 연산을 그래프에 추가해야 합니다.
init_op = tf.initialize_all_variables()

# 그래프를 올리고 연산을 실행합니다.
with tf.Session() as sess:
  # 'init' 연산을 실행합니다.
  sess.run(init_op)
  # 'state'의 초기값을 출력합니다.
  print(sess.run(state))
  # 'state'를 갱신하는 연산을 실행하고 'state'를 출력합니다.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))