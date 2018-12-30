from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from fire_detection_with_vgg19 import data_load

"""
<Error>
ValueError: Negative dimension size caused by subtracting 2 from 1 for 'max_pooling2d_2/MaxPool' (op: 'MaxPool') with input shapes: [?,1,112,128].

<Solve>
Quoting an answer mentioned in github, you need to specify the dimension ordering:

Keras is a wrapper over Theano or Tensorflow libraries. Keras uses the setting variable image_dim_ordering to decide if the input layer is Theano or Tensorflow format. This setting can be specified in 2 ways -

specify 'tf' or 'th' in ~/.keras/keras.json like so -  image_dim_ordering: 'th'. Note: this is a json file.
or specify the image_dim_ordering in your model like so: model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
Appendix: image_dim_ordering in 'th' mode the channels dimension (the depth) is at index 1 (e.g. 3, 256, 256). In 'tf' mode is it at index 3 (e.g. 256, 256, 3). Quoting @naoko from comments.

* model.add(MaxPooling2D((2,2), strides=(2,2)) ==> model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

<Reference>
https://github.com/fchollet/keras/issues/3945
https://stackoverflow.com/questions/39815518/keras-maxpooling2d-layer-gives-valueerror

"""

"""
<Error>
ImportError: `load_weights` requires h5py.

<Solve>
Have you tried directly installing h5py? http://docs.h5py.org/en/latest/build.html

Try running pip install h5py

* pip install h5py

<Reference>
https://github.com/fchollet/keras/issues/3426

"""


def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    train_fire_data = data_load.load_train_fire_data()
    train_fire_data = np.transpose(train_fire_data, (0, 3, 1, 2))
    train_fire_class = data_load.load_train_fire_result()

    train_nonfire_data = data_load.load_train_nonfire_data()
    train_nonfire_data = np.transpose(train_nonfire_data, (0, 3, 1, 2))
    train_nonfire_class = data_load.load_train_nonfire_result()

    test_fire_data = data_load.load_test_fire_data()
    test_fire_data = np.transpose(test_fire_data, (0, 3, 1, 2))
    test_fire_class = data_load.load_test_fire_result()
    test_nonfire_data = data_load.load_test_nonfire_data()
    test_fire_data = np.transpose(test_nonfire_data, (0, 3, 1, 2))
    test_nonfire_class = data_load.load_test_nonfire_result()

    print(train_fire_data.shape)
    print(train_fire_class.shape)
    print(train_nonfire_data.shape)
    print(train_nonfire_class.shape)

    # Test pretrained model
    #model = VGG_19('vgg19_weights.h5')
    model = VGG_19()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    model.fit(x=train_fire_data, y=train_fire_class, batch_size=100, epochs=10, verbose=1)
    model.fit(x=train_nonfire_data, y=train_nonfire_class, batch_size=100, epochs=10, verbose=1)

    test_fire_predict = model.predict(x=test_fire_data, batch_size=100)
    test_nonfire_predict = model.predict(x=test_nonfire_data, batch_size=100)

    #out = model.predict(im)
    #print(np.argmax(out))