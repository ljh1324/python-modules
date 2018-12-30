import tensorflow as tf
import functools, operator

def getLength(t):
    temp = (dim.value for dim in t.get_shape())         # dim is Dimension class.
                                                         # tensorflow 객체의 모양을 튜플로 만들어서 temp에 저장한다.
    return functools.reduce(operator.mul, temp)         # temp에 있는 값들을 모두 곱해준후 리턴한다.

def showConstant(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    sess.close()

def showConstantDetail(t):
    sess = tf.InteractiveSession()
    print(t.eval())
    print('shape :', tf.shape(t))
    print('size  :', tf.size(t))                        # 원소 개수 출력
    print('rank  :', tf.rank(t))                        # 차원 출력
    print(t.get_shape())

    sess.close()

def showVariable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    print(v.eval())
    sess.close()

def var2Numpy(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    n = v.eval()
    sess.close()

    return n

def op2Numpy(op):
    sess = tf.InteractiveSession()
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)
    ret = sess.run(op)
    sess.close()

    return ret

def showOperation(op):
    print(op2Numpy(op))


# 출처: http://pythonkim.tistory.com/62