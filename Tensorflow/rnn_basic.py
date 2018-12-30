import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

"""
tf.contrib.rnn.BasicRNNCell
tf.contrib.rnn.BasicLSTMCell
tf.contrib.rnn.GRUCell
tf.contrib.rnn.LSTMCell
tf.contrib.rnn.LayerNormBasicLSTMCell
"""

char_rdic = ['h', 'e', 'l', 'o'] # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)} # char -> id

"""

int	int_
bool            bool_
float           float_
complex         cfloat
str             string
unicode         unicode_
buffer          void
(all others)    object_

>>> dt = np.dtype(float)   # Python-compatible floating-point number
>>> dt = np.dtype(int)     # Python-compatible integer
>>> dt = np.dtype(object)  # Python object

"""
x_data = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0]], dtype=np.dtype(float))

sample = [char_dic[c] for c in "hello"] # to index
print(sample)

# Confuguration
char_vocab_size = len(char_dic)     # 4
rnn_size = char_vocab_size          # 1 hot coding (one of 4)
print("rnn_size: ", rnn_size)
time_step_size = 4                  # 'hell' -> predict 'ello'
batch_size = 1                      # one sample

# RNN model
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(x_data, time_step_size, 0)
print(X_split)

# output and next state
#outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, X_split, state)
outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, X_split, state, dtype=tf.float64)
# logits: list of 2D Tensors of shape [batch_size_x num_decoder_symbols].
# targets: list of 1D batch-sized int32 Tensors of the same length as logits.
# weight: list of 1D batch-sized float-Tensors of the same length as logis
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss)  / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print(result, [char_rdic[t] for t in result])

