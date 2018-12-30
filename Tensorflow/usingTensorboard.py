# 텐서플로 입문 책 p.48 참조

import tensorflow as tf

a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a+b*2, name="y")

model = tf.initialize_all_variables()
with tf.Session() as session:
    merged = tf.summary.merge_all
    writer = tf.summary.FileWriter("./tensorflowlogs", session.graph)       # 실행 그래에 존재하는 모든 요약 명령어 결과값을 기록한다.
                                                                            # p.49를 참고하여 데이터 플로우 그래프와 보조 노드를 확인해보자.
                                                                            # ex) tensorboard --logdir=D:\MyPythonProject\Tensorflow\tensorflowlogs
    session.run(model)
    print(session.run(y))

