from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import os
import numpy as np


def small_vgg_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(512, 512, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def save_model(file_path, num, data_set, model):
    save_model = Sequential()

    idx = 1
    for layer in model.layers:
        dir_path = os.path.join(file_path, 'layer ' + str(idx))
        print(dir_path)
        os.makedirs(dir_path)
        f_write = open(os.path.join(dir_path, num))

        save_model.add(layer)
        result = save_model.predict(data_set)
        print(result.shape)


if __name__ == "__main__":
    model = small_vgg_19()
    model_info = ""
    for layer in model.layers:
        print("layer: ", layer, end=':')
        print("weight: ", layer.get_weights())
        print("output: ", layer.output)

    save_model('test', 1,  model)
    print()
    for layer in model.layers:
        print(layer.output_shape)



