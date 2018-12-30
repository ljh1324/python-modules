import tensorflow as tf
import functions

c2, c3 = tf.constant([1.2, 5.6]), tf.constant([-4, -1, 7])
v2, v3 = tf.Variable([2.3, 4.5]), tf.Variable([-2,  3, 5])

print('-----------add_n------------')           # same shape aod dtype 덧셈   # 열 덧셈
functions.showOperation(tf.add_n([c2, v2]))      # [  3.5         10.10000038]
functions.showOperation(tf.add_n([c3, v3, v3]))  # [-8  5 17]

print('-----------abs------------')
functions.showOperation(tf.abs(c2))              # [ 1.20000005  5.5999999 ]
functions.showOperation(tf.abs([c3, v3]))        # [[4 1 7] [2 3 5]]

print('-----------neg------------')
functions.showOperation(tf.negative(c2))              # [-1.20000005 -5.5999999 ]
functions.showOperation(tf.negative([c3, v3]))        # [[ 4  1 -7] [ 2 -3 -5]]

print('-----------sign------------')
functions.showOperation(tf.sign(c2))             # [ 1.  1.]
functions.showOperation(tf.sign([c3, v3]))       # [[-1 -1  1] [-1  1  1]]

print('-----------square------------')
functions.showOperation(tf.square(c2))           # [  1.44000006  31.3599987 ]
functions.showOperation(tf.square([c3, v3]))     # [[16  1 49] [ 4  9 25]]

print('-----------round------------')
functions.showOperation(tf.round(c2))            # [ 1.  6.]
functions.showOperation(tf.round([c3, v3]))      # [[-4 -1  7] [-2  3  5]]

print('-----------sqrt------------')            # 정수 사용 불가. 음수는 nan. 루트
functions.showOperation(tf.sqrt(c2))             # [ 1.09544516  2.36643195]
functions.showOperation(tf.sqrt([c2, v2]))       # [[ 1.09544516  2.36643171]
                                                 # [ 1.5165751   2.12132025]]

c2, c3 = tf.constant([1.2, 5.6]), tf.constant([-4, -1, 7])
v2, v3 = tf.Variable([2.3, 4.5]), tf.Variable([-2,  3, 5])

print('-----------pow------------')             # 갯수 일치 주의
functions.showOperation(tf.pow(c2, 2))           # [  1.44000006  31.3599987 ]
# [[   1.44000006  175.61599731] [   5.28999996   91.125     ]]
functions.showOperation(tf.pow([c2, v2], [2, 3]))

functions.showOperation(tf.pow(c2, [1, 3]))         # [   1.20000005  175.61599731]
functions.showOperation(tf.pow(v2, [1, 2]))         # [   2.29999995  20.25      ]

# [[   1.20000005  175.61599731] [   2.29999995   20.25      ]]
functions.showOperation(tf.pow([c2, v2], [[1, 3], [1, 2]]))

print('-----------exp------------')             # 정수 사용 불가. 음수 동작. e ^ x
functions.showOperation(tf.exp(c2))              # [   3.320117   270.4263916]
functions.showOperation(tf.exp([c2, v2]))        # [[   3.320117    270.4263916 ] [   9.97418213   90.01712799]]

print('-----------log------------')             # 정수 사용 불가. 음수는 nan. 밑이 e인 log
functions.showOperation(tf.log(c2))              # [ 0.18232159  1.72276664]
functions.showOperation(tf.log([c2, v2]))        # [[ 0.18232159  1.72276664] [ 0.83290911  1.50407743]]

print('-----------ceil------------')            # 정수 사용 불가.
functions.showOperation(tf.ceil(c2))             # [ 2.  6.]
functions.showOperation(tf.ceil([c2, v2]))       # [[ 2.  6.] [ 3.  5.]]

print('-----------floor------------')           # 정수 사용 불가.
functions.showOperation(tf.floor(c2))            # [ 1.  5.]
functions.showOperation(tf.floor([c2, v2]))      # [[ 1.  5.] [ 2.  4.]]

print('-----------maximum------------')
functions.showOperation(tf.maximum(c2, v2))      # [ 2.29999995  5.5999999 ]    열로 비교해서 최댓값
# [[ 2.29999995  5.5999999 ] [ 2.29999995  5.5999999 ]]. [c2, c3] is error because shape is different.
functions.showOperation(tf.maximum([c2, c2], [v2, v2]))

print('-----------minimum------------')
functions.showOperation(tf.minimum(c2, v2))      # [ 1.20000005  4.5       ]
# [[ 1.20000005  4.5       ] [ 1.20000005  4.5       ]]
functions.showOperation(tf.minimum([c2, c2], [v2, v2]))

print('-----------sin------------')             # 정수 사용 불가.
functions.showOperation(tf.sin(c2))              # [ 0.93203908 -0.63126671]
# [[ 0.93203908 -0.63126671] [ 0.74570525 -0.97753012]]
functions.showOperation(tf.sin([c2, v2]))

print('-----------cos------------')             # 정수 사용 불가.
functions.showOperation(tf.cos(c2))              # [ 0.36235771  0.7755658 ]
# [[ 0.36235774  0.77556586] [-0.66627598 -0.21079579]]
functions.showOperation(tf.cos([c2, v2]))

print('-----------tan------------')             # 정수 사용 불가.
functions.showOperation(tf.tan(c2))              # [ 2.5721519  -0.81394345]
# [[ 2.5721519  -0.81394345] [-1.1192137   4.63733196]]
functions.showOperation(tf.tan([c2, v2]))

c4 = tf.constant([[1, 3, 5], [0, 2, 4]])
v4 = tf.Variable([[1, 2], [3, 7], [8, 9]])

print('-----------matmul------------')          # (2x3)*(3x2) = (2x2), (3x2)*(2x3) = (3x3)
functions.showOperation(tf.matmul(c4, v4))       # [[50 68] [38 50]]
functions.showOperation(tf.matmul(v4, c4))       # [[ 1  7 13] [ 3 23 43] [ 8 42 76]]
