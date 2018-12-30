from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from fire_detection_with_unet.data_input import load_train_data_idx, load_test_data_idx
from fire_detection_with_unet.data_input import load_train_data, load_test_data


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 224
img_cols = 224

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    print(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    print(conv1);
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    print(conv1);
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1);
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    print(conv2);
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    print(conv2);
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    print(conv3);
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    print(conv3);
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    print(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    print(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    print(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    print(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    print(conv5)
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    print(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    print(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    print(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    print(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    print(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    print(conv7)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    print(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    print(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    print(conv8)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    print(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    print(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    print(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    print(conv10)
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])    # metrics: 출력할 데이터

    return model


# 축을 하나 늘린다.
def preprocess(imgs):
    # image 개수, image row, image col 만큼 할당
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)  # imgs[i]를 img_cols, img_rows에 맞게 조절

    imgs_p = imgs_p[..., np.newaxis]    # 이미지에 새로운 축 추가(아마 RGB값을 나누는 용도 인듯)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #imgs_train, imgs_mask_train = load_train_data_idx(1)
    imgs_train, imgs_mask_train = load_train_data()
    print(imgs_train.shape)
    imgs_train = preprocess(imgs_train)
    print('Image Train Processiong Done...')
    imgs_mask_train = preprocess(imgs_mask_train)
    print('Image Train Mask Processiong Done...')

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  # nomalization
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=10, nb_epoch=1000, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    model.save_weights("weights.h5")

    print('-'*30)
    print('Loading and preprocessing happy_song data...')
    print('-'*30)
    # imgs_test, imgs_id_test = load_test_data_idx(1)          # image파일과 id파일을 불러온다.
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on happy_song data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    idx = 0
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        print(image_id)
        imsave(os.path.join(pred_dir, str(idx) + '_pred.png'), image)
        idx += 1


if __name__ == '__main__':
    #get_unet()
    train_and_predict()
