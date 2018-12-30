"""
 <etting started with the Keras Sequential model>

The Sequential model is a linear stack of layers.

You can create a Sequential model by passing a list of layer instances to the constructor:

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

You can also simply add layers via the .add() method:
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

 <Specifying the input shape>
input_shape : 인티저 타입 형식의 튜플, None일 경우 어떠한 인풋값이 되든 상관없는 듯
input_dim : Dense로 층을 하나 추가할 때 input값의 크기를 지정하는데 사용된다(2D Layer)
batch_size : batch_size를 고정해야할 필요가 있을 경우 batch_size를 이용하여 고정한다. 만약 batch_size=32 이고 input_shape=(6, 8) 일때 (32, 6, 8)의 고정된 크기로 입력받는다.

 * 다음 두 코드는 똑같은 결과를 만든다.
 1.
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

 2.
model = Sequential()
model.add(Dense(32, input_dim=784))

 < Compilation >
모델을 트레이닝 하기전, learning process를 설정해줘야 한다. 이것은 "compile" 메소드를 이용하면 된다.
"compile"은 3가지 argument를 받는다.
 - optimizer : 기본적으로 제공하는 rmsprop, adagrad optimizer가 있는데 이는 Optimizer 클래스의 인스턴스이다.
 - loss function : 모델이 이 loss function을 최소화 하도록 한다. 존재하는 loss function에는 "categorical_crossentropy"와 "mse" 가 있다.
 - metrics : fit을 통해 학습을 시킬때, 추가로 출력할 값을 지정할 수 있다. metrics=['accuracy'] 가 기본적으로 제공된다.

 < Training >
 - fit : 모델을 트레이닝 시키는데 사용한다.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

model = Sequential()                                                # The Sequential model is a linear stack of layers.

# keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
#                         bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
#                         activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
model.add(Dense(input_dim=3, units=1))                              # input shape 설정
model.add(Activation('linear'))                                     # Activation 설정

rmsprop = RMSprop(lr=1e-10)     # learning rate 설정.
model.compile(loss='mse', optimizer=rmsprop, metrics=['accuracy'])
model.fit(x_data, y_data, epochs=1000)                              # 1000번 반복

y_predict = model.predict(np.array([[95., 100., 80]]))              # 만들어진 model에 데이터를 넣어서 결과값을 만듦
print(y_predict)

# ----------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np

# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))              # model take as input array of shape(*, 100)
                                                                    # and output arrays of shape (*, 32)
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

# ----------------------------------------------------------------------------------------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
# label이 0 ~ 9 까지의 데이터인데 이것을 one-hot encoding 시켜주었다.
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

print(labels)
print(one_hot_labels)
