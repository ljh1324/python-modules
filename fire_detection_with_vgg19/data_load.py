import os
from skimage.io import imsave, imread
from skimage.transform import resize, downscale_local_mean
from skimage.viewer import ImageViewer

import numpy as np
import cv2
import fire_detection_with_vgg19.hog_function as myhog
import matplotlib.pyplot as plt

# for vgg19
# data_path = 'data'

fix_image_row = 224
fix_image_col = 224


def create_fire_data(data_path):
    # data\\fire
    train_data_path = os.path.join(data_path, 'fire')      # data_path와 train데이터를 잇는다.
    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    total = len(images)

    imgs = np.ndarray((total, fix_image_row, fix_image_col, 3), dtype=np.uint8) # An array object represents a multidimensional, homogeneous array of fixed-size items.
    img_values = [[1, 0] for i in range(total)]

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:

        # imread(fname, as_grey=False, plugin=None, flatten=None, **plugin_args)
        """
        Parameters:
            fname : string
                Image file name, e.g. happy_song.jpg or URL.
            as_grey : bool
                If True, convert color images to grey-scale (64-bit floats). Images that are already in grey-scale format are not converted.
            plugin : str
                Name of plugin to use. By default, the different plugins are tried (starting with the Python Imaging Library) until a suitable candidate is found.
                If not given and fname is a tiff file, the tifffile plugin will be used.
        Returns:
            img_array : ndarray
                The different color bands/channels are stored in the third dimension, such that a grey-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
        """

        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        #viewer = ImageViewer(img)
        #viewer.show()

        img_resize = resize(img, (fix_image_row, fix_image_col), mode='reflect')
        img_resize = np.array([img_resize])
        #print(img_resize.shape[0], img_resize.shape[1], img_resize.shape[2], img_resize.shape[3])
        imgs[i] = img_resize

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path, 'imgs_train_fire.npy'), imgs)
    np.save(os.path.join(data_path, 'imgs_values_fire'), img_values)
    print('Saving to .npy files done.')

def create_nonfire_data(data_path):
    # data\\street
    train_data_path = os.path.join(data_path, 'nonfire')  # data_path와 train데이터를 잇는다.
    images = os.listdir(train_data_path)  # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    total = len(images)

    imgs = np.ndarray((total, fix_image_row, fix_image_col, 3),
                      dtype=np.uint8)  # An array object represents a multidimensional, homogeneous array of fixed-size items.
    img_values = [[0, 1] for i in range(total)]

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        img = imread(os.path.join(train_data_path, image_name), as_grey=False)
        #print(img)
        img_resize = resize(img, (fix_image_row, fix_image_col), mode='reflect')
        img_resize = np.array([img_resize])
        imgs[i] = img_resize

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path, 'imgs_train_nonfire.npy'), imgs)
    np.save(os.path.join(data_path, 'imgs_values_nonfire.npy'), img_values)
    print('Saving to .npy files done.')

def create_fire_hog_data(data_path):
    train_data_path = os.path.join(data_path, 'fire_bright')
    images = os.listdir(train_data_path)
    print(images)
    total = len(images)

    img = cv2.cvtColor(cv2.imread(os.path.join(train_data_path, images[0])), cv2.COLOR_BGR2GRAY)
    gradients = myhog.hog(img)
    print(gradients.shape[0], gradients.shape[1], gradients.shape[2])

    imgs = np.ndarray((total, gradients.shape[0], gradients.shape[1], gradients.shape[2]),
                      dtype=np.uint8)  # An array object represents a multidimensional, homogeneous array of fixed-size items.
    img_values = [[1, 0] for i in range(total)]

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(train_data_path, image_name)), cv2.COLOR_BGR2GRAY)
        gradients = myhog.hog(img)
        imgs[i] = gradients

        """
        bin = 8  # angle is 360 / nbins * direction
        plt.pcolor(gradients[:, :, bin])  # 특정 bin의 값 출력.
        plt.gca().invert_yaxis()  # y축 invert(뒤집기)
        plt.gca().set_aspect('equal', adjustable='box')  # 전체크기랑 똑같이 맞춘다. 없을 경우 정사각형
        plt.colorbar()
        plt.show()
        """

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path, 'imgs_train_hog_fire.npy'), imgs)
    np.save(os.path.join(data_path, 'imgs_values_hog_fire.npy'), img_values)
    print('Saving to .npy files done.')

def create_nonfire_hog_data(data_path):
    train_data_path = os.path.join(data_path, 'nonfire_bright')
    images = os.listdir(train_data_path)
    print(images)
    total = len(images)

    img = cv2.cvtColor(cv2.imread(os.path.join(train_data_path, images[0])), cv2.COLOR_BGR2GRAY)
    gradients = myhog.hog(img)
    print(gradients.shape[0], gradients.shape[1], gradients.shape[2])

    imgs = np.ndarray((total, gradients.shape[0], gradients.shape[1], gradients.shape[2]),
                      dtype=np.uint8)  # An array object represents a multidimensional, homogeneous array of fixed-size items.
    img_values = [[0, 1] for i in range(total)]

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(train_data_path, image_name)), cv2.COLOR_BGR2GRAY)
        gradients = myhog.hog(img)

        """
        bin = 8  # angle is 360 / nbins * direction
        plt.pcolor(gradients[:, :, bin])  # 특정 bin의 값 출력.
        plt.gca().invert_yaxis()  # y축 invert(뒤집기)
        plt.gca().set_aspect('equal', adjustable='box')  # 전체크기랑 똑같이 맞춘다. 없을 경우 정사각형
        plt.colorbar()
        plt.show()
        """
        imgs[i] = gradients

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_path, 'imgs_train_hog_nonfire.npy'), imgs)
    np.save(os.path.join(data_path, 'imgs_values_hog_nonfire.npy'), img_values)
    print('Saving to .npy files done.')


# load train data function
def load_train_fire_data():
    return np.load(r'data\train\imgs_train_fire.npy')


def load_train_fire_result():
    return np.load(r'data\train\imgs_values_fire.npy')


def load_train_nonfire_data():
    return np.load(r'data\train\imgs_train_nonfire.npy')


def load_train_nonfire_result():
    return np.load(r'data\train\imgs_values_nonfire.npy')


def load_train_hog_fire():
    return np.load(r'data\train\imgs_train_hog_fire.npy')


def load_train_hog_fire_result():
    return np.load(r'data\train\imgs_values_hog.npy')


def load_train_hog_nonfire():
    return np.load(r'data\train\imags_train_hog_nonfire.npy')


def load_train_hog_nonfire_result():
    return np.load(r'data\train\imgs_values_hog_nonfire.npy')


# load happy_song data function
def load_test_fire_data():
    return np.load(r'data\test\imgs_train_fire.npy')


def load_test_fire_result():
    return np.load(r'data\test\imgs_values_fire.npy')


def load_test_nonfire_data():
    return np.load(r'data\test\imgs_train_nonfire.npy')


def load_test_nonfire_result():
    return np.load(r'data\test\imgs_values_nonfire.npy')


def load_test_hog_fire():
    return np.load(r'data\test\imgs_train_hog_fire.npy')


def load_test_hog_fire_result():
    return np.load(r'data\test\imgs_values_hog.npy')


def load_test_hog_nonfire():
    return np.load(r'data\test\imgs_train_hog_nonfire.npy')


def load_test_hog_nonfire_result():
    return np.load(r'data\test\imgs_values_hog_nonfire.npy')


if __name__ == '__main__':
    train_data_path = 'data\\train'
    test_data_path = 'data\\happy_song'

    create_fire_data(train_data_path)
    create_nonfire_data(train_data_path)
    create_fire_data(test_data_path)
    create_nonfire_data(test_data_path)

    create_fire_hog_data(train_data_path)
    create_nonfire_hog_data(train_data_path)
    create_fire_hog_data(test_data_path)
    create_nonfire_hog_data(test_data_path)

