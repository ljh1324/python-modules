from __future__ import print_function

import os
import numpy as np

from skimage.transform import resize
from skimage.io import imsave, imread

# for unet

data_path = 'data/'

image_rows = 96
image_cols = 96


def create_train_data():
    train_data_path = os.path.join(data_path, 'only_fire')      # data_path와 train데이터를 잇는다.
    train_mask_data_path = os.path.join(data_path, 'only_fire_bright')

    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    images_mask = os.listdir(train_mask_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8) # An array object represents a multidimensional, homogeneous array of fixed-size items.
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        #image_mask_name = image_name.split('.')[0] + '_mask.tif'            # 저장할 파일 정보형식을 새로 만든다.
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_resize = np.array([img_resize])               # np.array 형태로 변형.
        imgs[i] = img_resize

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    for image_name in images_mask:
        img = imread(os.path.join(train_mask_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_mask = np.array([img_resize])
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(data_path, 'my_image')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating happy_song images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])      # image파일의 이름만 때내어 변수에 저장
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])
        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
