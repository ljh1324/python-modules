import tensorflow as tf
import matplotlib.pyplot as plt

norm = tf.random_normal([100], mean = 0, stddev = 2)        # 평균값 = 0, 표준 편차 = 2인 정규 분포를 따르는 크기가 100인 1차원 텐서 정의
with tf.Session() as session:
    plt.hist(norm.eval())                      # eval을 통한 값 평가, plt.hist를 통한 histogram 그리기
    #plt.hist(norm.eval(), normed=True)
    plt.show()
