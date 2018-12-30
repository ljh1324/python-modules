import tensorflow as tf
import functions

c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.constant([1, 3, 5])
v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
v2 = tf.constant([[1, 2, 3], [7, 8, 9]])

print('-----------slice------------')
functions.showOperation(tf.slice(c1, [2], [3]))             # [5 7 9]
functions.showOperation(tf.slice(v1, [0, 2], [1, 2]))       # [[3 4]]                   # 0행 2열 - 1행 4열
functions.showOperation(tf.slice(v1, [0, 2], [2, 2]))       # [[3 4] [9 0]]             # 0행 2열 - 2행 4열
functions.showOperation(tf.slice(v1, [0, 2], [2,-1]))       # [[3 4 5 6] [9 0 1 2]]     # 0행 2열 - 2행 끝열

print('-----------split------------')
#functions.showOperation(tf.split(0, 2, c1)) # [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]]
functions.showOperation(tf.split(c1, 2, 0))     # 0: 한단계 들어가서(scalar), 2: 2개로 나눔
#functions.showOperation(tf.split(0, 5, c1)) # [[1, 3], [5, 7], [9, 0], [2, 4], [6, 8]]
functions.showOperation(tf.split(c1, 5, 0))     # 0: 한단계 들어가서(scalar), 5: 5개로 나눔
#functions.showOperation(tf.split(0, 2, v1)) # [[[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 0, 1, 2]]]
functions.showOperation(tf.split(v1, 2, 0))     # 0: 한단계 들어가서(1-rank), 2: 2개로 나눔
#functions.showOperation(tf.split(1, 2, v1))    # [[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [0, 1, 2]]]
functions.showOperation(tf.split(v1, 2, 1))     # 1: 두단계 들어가서(2-rank), 2: 2개로 나눔

print('-----------tile------------')
functions.showOperation(tf.tile(c2, [3]))   # [1 3 5 1 3 5 1 3 5]
# [[1 2 3 1 2 3] [7 8 9 7 8 9] [1 2 3 1 2 3] [7 8 9 7 8 9]]
functions.showOperation(tf.tile(v2, [2, 2]))
# [[1 2 3 1 2 3 1 2 3] [7 8 9 7 8 9 7 8 9]
#  [1 2 3 1 2 3 1 2 3] [7 8 9 7 8 9 7 8 9]]
functions.showOperation(tf.tile(v2, [2, 3]))

print('-----------pad------------')         # 2차원에 대해서만 동작
# [[0 0 0 0 0 0 0]
#  [0 0 1 2 3 0 0]
#  [0 0 7 8 9 0 0]
#  [0 0 0 0 0 0 0]]
functions.showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'CONSTANT'))
# [[9 8 7 8 9 8 7]
#  [3 2 1 2 3 2 1]
#  [9 8 7 8 9 8 7]
#  [3 2 1 2 3 2 1]]     # 3 2 1 2 3 2 1 2 3 2 1 처럼 반복
functions.showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'REFLECT'))
# [[2 1 1 2 3 3 2]
#  [2 1 1 2 3 3 2]
#  [8 7 7 8 9 9 8]
#  [8 7 7 8 9 9 8]]     # 3 2 1 (1 2 3) 3 2 1. 가운데와 대칭
functions.showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'SYMMETRIC'))
functions.showOperation(tf.pad(v2, [[2, 3], [3, 2]], 'CONSTANT'))             # 위쪽에 2칸 아래쪽에 3칸, 왼쪽에 3칸, 오른쪽에 2칸을 padding한다
#functions.showOperation(tf.pad(v2, [[1, 1], [5, 5]], 'REFLECT'))
#functions.showOperation(tf.pad(v2, [[1, 1], [5, 5]], 'SYMMETRIC'))

# functions.showOperation(tf.pad(v2, [[2, 3], [2, 3]], 'REFLECT'))            # 좌우 대칭이 안되어서 에러, 위 아래로도 안된다.
# functions.showOperation(tf.pad(v2, [[2, 3], [2, 3]], 'SYMMETRIC'))          # 좌우 대칭, 위아래 대칭이 안된다

print('-----------concat------------')
#functions.showOperation(tf.concat(0, [c1, c2]))     # [1 3 5 7 9 0 2 4 6 8 1 3 5]
functions.showOperation(tf.concat([c1, c2], 0))
#functions.showOperation(tf.concat(1, [v1, v2]))     # [[1 2 3 4 5 6 1 2 3] [7 8 9 0 1 2 7 8 9]]
functions.showOperation(tf.concat([v1, v2], 1))
# functions.showOperation(tf.concat(0, [v1, v2]))   # error. different column size.

c3, c4 = tf.constant([1, 3, 5]), tf.constant([[1, 3, 5], [5, 7, 9]])
v3, v4 = tf.constant([2, 4, 6]), tf.constant([[2, 4, 6], [6, 8, 0]])

print('-----------reverse------------')
# c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
# v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
functions.showOperation(tf.reverse(c1, [0]))         # [8 6 4 2 0 9 7 5 3 1]
functions.showOperation(tf.reverse(v1, [0]))  # [[7 8 9 0 1 2] [1 2 3 4 5 6]]
functions.showOperation(tf.reverse(v1, [1]))  # [[6 5 4 3 2 1] [2 1 0 9 8 7]]

print('-----------transpose------------')      # perm is useful to multi-dimension .
functions.showOperation(tf.transpose(c3))       # [1 3 5]. not 1-D.
functions.showOperation(tf.transpose(c4))       # [[1 5] [3 7] [5 9]]
functions.showOperation(tf.transpose(c4, perm=[0, 1]))   # [[1 3 5] [5 7 9]]
functions.showOperation(tf.transpose(c4, perm=[1, 0]))   # [[1 5] [3 7] [5 9]]

print('-----------gather------------')
functions.showOperation(tf.gather(c1, [2, 5, 2, 5]))     # [5 0 5 0]    # c1의 2번째, 5번째 원소를 모음
functions.showOperation(tf.gather(v1, [0, 1]))           # [[1 2 3 4 5 6] [7 8 9 0 1 2]]
functions.showOperation(tf.gather(v1, [[0, 0], [1, 1]])) # [[[1 2 3 4 5 6] [1 2 3 4 5 6]]  [[7 8 9 0 1 2] [7 8 9 0 1 2]]]

print('-----------one_hot------------')         # make one-hot matrix.
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]
#  [ 0.  1.  0.]]
functions.showOperation(tf.one_hot([0, 1, 2, 1], 3))        #   [0열에 1, 1열에 1, 2열에 1, 1열에 1]
# [[ 0.  0.  0.  1.]
#  [ 0.  0.  0.  0.]
#  [ 0.  1.  0.  0.]]
functions.showOperation(tf.one_hot([3, -1, 1], 4))
