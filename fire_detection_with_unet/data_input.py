from __future__ import print_function

import os
import numpy as np

from skimage.transform import resize
from skimage.io import imsave, imread

# for unet

image_rows = 224
image_cols = 224


def create_train_data(parent_dir, img_dir, img_mask_dir, save_img_file, save_img_mask_file):
    train_data_path = os.path.join(parent_dir, img_dir)      # data_path와 train데이터를 잇는다.
    train_mask_data_path = os.path.join(data_path, img_mask_dir)

    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    images_mask = os.listdir(train_mask_data_path)
    total = len(images)
    print(total)
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8) # An array object represents a multidimensional, homogeneous array of fixed-size items.
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        # image_mask_name = image_name.split('.')[0] + '_mask.tif'            # 저장할 파일 정보형식을 새로 만든다.
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_resize = np.array([img_resize])               # np.array 형태로 변형.
        imgs[i] = img_resize

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    i = 0
    for image_name in images_mask:
        img = imread(os.path.join(train_mask_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_mask = np.array([img_resize])
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')

    np.save(os.path.join(parent_dir, save_img_file), imgs)
    np.save(os.path.join(parent_dir, save_img_mask_file), imgs_mask)
    print('Saving to .npy files done.')


def create_train_divide_data(parent_dir, img_dir, img_mask_dir, save_img_file, save_img_mask_file, divide_size):
    train_data_path = os.path.join(parent_dir, img_dir)      # data_path와 train데이터를 잇는다.
    train_mask_data_path = os.path.join(data_path, img_mask_dir)

    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    images_mask = os.listdir(train_mask_data_path)
    total = len(images)
    print(total)
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8) # An array object represents a multidimensional, homogeneous array of fixed-size items.
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    file_count = 0
    divide_count = 1
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        # image_mask_name = image_name.split('.')[0] + '_mask.tif'            # 저장할 파일 정보형식을 새로 만든다.
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_resize = np.array([img_resize])               # np.array 형태로 변형.
        imgs[i] = img_resize

        i += 1
        file_count += 1
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(file_count, total))
        if i != 0 and i % divide_size == 0:
            np.save(os.path.join(parent_dir, '%s%d.npy' % (save_img_file, divide_count)), imgs)
            divide_count += 1
            i = 0

    if i > 0:
        np.save(os.path.join(parent_dir, '%s%d.npy' % (save_img_file, divide_count)), imgs)

    i = 0
    file_count = 0
    divide_count = 1
    print('-'*30)
    print('Creating training mask images...')
    print('-'*30)
    for image_name in images_mask:
        img = imread(os.path.join(train_mask_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_mask = np.array([img_resize])
        imgs_mask[i] = img_mask

        i += 1
        file_count += 1
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(file_count, total))
        if i != 0 and i % divide_size == 0:
            np.save(os.path.join(parent_dir, '%s%d.npy' % (save_img_mask_file, divide_count)), imgs_mask)
            divide_count += 1
            i = 0

    print('Loading done.')

    if i > 0:
        np.save(os.path.join(parent_dir, '%s%d.npy' % (save_img_mask_file, divide_count)), imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('data\\unet\\img_result.npy')
    imgs_mask_train = np.load('data\\unet\\img_result_mask.npy')

    #imgs_train = np.load('data\\unet\\img_train.npy')
    #imgs_mask_train = np.load('data\\unet\\img_train_mask.npy')
    return imgs_train, imgs_mask_train


def load_train_data_idx(idx):
    data_dir = 'data\\unet'
    img_file_name = 'img_train'
    img_mask_file_name = 'img_train_mask'

    imgs_train = np.load(os.path.join(data_dir, '%s%d.npy' % (img_file_name, idx)))
    imgs_mask_train = np.load(os.path.join(data_dir, '%s%d.npy' % (img_mask_file_name, idx)))

    return imgs_train, imgs_mask_train


def create_test_data(parent_dir, img_dir, save_img_file, save_id_file):
    train_data_path = os.path.join(parent_dir, img_dir)
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=str)

    i = 0
    print('-'*30)
    print('Creating happy_song images...')
    print('-'*30)
    for image_name in images:
        img_id = image_name.split('.')[0]      # image파일의 이름만 때내어 변수에 저장
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_resize = np.array([img_resize])
        imgs[i] = img_resize
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(parent_dir, save_img_file), imgs)
    np.save(os.path.join(parent_dir, save_id_file), imgs_id)
    print('Saving to .npy files done.')


def create_test_divide_data(parent_dir, img_dir, save_img_file, save_id_file, divide_size):
    train_data_path = os.path.join(parent_dir, img_dir)
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=str)

    i = 0
    file_count = 0
    divide_count = 1
    print('-'*30)
    print('Creating happy_song images...')
    print('-'*30)
    for image_name in images:
        img_id = image_name.split('.')[0]      # image파일의 이름만 때내어 변수에 저장
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_resize = resize(img, (image_rows, image_cols), mode='reflect')
        img_resize = np.array([img_resize])
        imgs[i] = img_resize
        imgs_id[i] = img_id

        i += 1
        file_count += 1
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(file_count, total))
        if i != 0 and i % divide_size == 0:
            np.save(os.path.join(parent_dir, '%s%d.npy' % (save_img_file, divide_count)), imgs)
            np.save(os.path.join(parent_dir, '%s%d.npy' % (save_id_file, divide_count)), imgs)
            divide_count += 1
            i = 0
            i %= divide_count

    print('Loading done.')

    if i > 0:
        np.save(os.path.join(parent_dir, '%s%d.npy' % (save_img_file, divide_count)), imgs)
        np.save(os.path.join(parent_dir, '%s%d.npy' % (save_id_file, divide_count)), imgs)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('data\\unet\\img_test_result.npy')
    imgs_id = np.load('data\\unet\\img_test_id.npy')
    #data_dir = 'data\\unet'
    #img_npy = 'img'
    #imgs_test = np.load('data\\unet\\img_test.npy')
    #imgs_id = np.load('data\\unet\\img_test_id.npy')
    return imgs_test, imgs_id

def load_test_data_idx(idx):
    #imgs_test = np.load('data\\unet\\img_test_result.npy')
    #imgs_id = np.load('data\\unet\\img_test_id.npy')

    data_dir = 'data\\unet'
    img_file_name = 'img_test'
    img_id_file_name = 'image_test_id'

    imgs_test = np.load(os.path.join(data_dir, '%s%d.npy' % (img_file_name, idx)))
    imgs_id = np.load(os.path.join(data_dir, '%s%d.npy' % (img_id_file_name, idx)))
    return imgs_test, imgs_id

if __name__ == '__main__':
    data_path = 'data\\unet'
    create_train_data(data_path, 'result', 'result_mask', 'img_result.npy', 'img_result_mask.npy')
    create_test_data(data_path, 'result', 'img_test_result.npy', 'img_test_id.npy')
    #create_train_divide_data(data_path, 'result', 'result_mask', 'img_train', 'img_train_mask', 100)
    #create_test_divide_data(data_path, 'result', 'img_test', 'image_test_id', 100)

