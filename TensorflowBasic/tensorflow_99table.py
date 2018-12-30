# 예제1 : 1부터 3까지 증가
# 예제2 : 구구단

import tensorflow as tf

def one2three_1():
    state = tf.Variable(1)
    one   = tf.constant(1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(3):
            print(sess.run(state))
            state = tf.add(state, one)

def one2three_2():
    state = tf.Variable(0)
    one   = tf.constant(1)
    value = tf.add(state, one)
    update = tf.assign(state, value)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(3):
            print(sess.run(update))

def table99_1(which):
    level = tf.constant(which)
    state = tf.Variable(1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, 10):
            left, rite = sess.run(level), sess.run(state)
            state = tf.add(state, 1)

            print('{} x {} = {:2}'.format(left, rite, left*rite))

def table99_2(which):
    # update를 처리하면 연계된 모든 Tensor 객체도 함께 처리됨
    level  = tf.constant(which)
    state  = tf.Variable(0)
    add    = tf.add(state, 1)
    value  = tf.assign(state, add)
    update = tf.mul(level, value)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(9):
            # left, rite, result = sess.run([level, state, update])

            result = sess.run(update)
            left, rite = level.eval(), state.eval()

            print('{} x {} = {:2}'.format(left, rite, result))

def table99_3(which):
    left = tf.placeholder(tf.int32)
    rite = tf.placeholder(tf.int32)
    update = tf.mul(left, rite)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1, 10):
            result = sess.run(update, feed_dict={left: which, rite: i})
            print('{} x {} = {:2}'.format(which, i, result))

one2three_1()
one2three_2()

table99_1(7)
table99_2(7)
table99_3(7)


# 출처: http://pythonkim.tistory.com/68 [파이쿵]