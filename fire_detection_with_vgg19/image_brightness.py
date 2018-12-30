from PIL import Image
import numpy as np
import copy
import os
import cv2

def rgb_to_ycbcr(rgb_value):
    for_dot_value = np.array([[0.299, -0.169, 0.5], [0.587, -0.331, -0.81], [0.114, 0.5, -0.81]])
    for_add_value = np.array([16, 128, 128])
    for_dot_rgb = np.array(rgb_value)
    return tuple(np.dot(for_dot_rgb, for_dot_value).astype(int) + for_add_value)

def to_ycbcr_img(img):
    img_ycbcr = copy.deepcopy(img)
    pix = img_ycbcr.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pix[i, j] = rgb_to_ycbcr(pix[i, j])
    return img_ycbcr

def get_ycbcr_mean_values(img):
    ycbcr_mean = np.zeros(3, int)
    size = img.size[0] * img.size[1]
    pix = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            ycbcr_mean += pix[i, j]

    return ycbcr_mean / size

def emphasis_brightness(img):
    img_ycbcr = to_ycbcr_img(img)
    ycbcr_mean = get_ycbcr_mean_values(img)
    y_mean = ycbcr_mean[0]
    cb_mean = ycbcr_mean[1]
    cr_mean = ycbcr_mean[2]
    pix = img_ycbcr.load()
    threshold = 70

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            y = pix[i, j][0]
            cb = pix[i, j][1]
            cr = pix[i, j][2]
            #print(y, cb, cr)
            #print(y_mean, cb_mean, cr_mean)
            # if (y > y_mean and cb > cb_mean and cr > cr_mean and abs(cb - cr) >= threshold):

            # 전체적인 이미지가 같은 색깔일 수 록 문제가 발생한다!!!!

            if y - y_mean > threshold:
                pix[i, j] = (255, 255, 255)
            else:
                pix[i, j] = (0, 0, 0)
    return img_ycbcr


def to_empty_picture(img):
    pix = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
                pix[i, j] = (0, 0, 0)
    return img

fix_image_size = 224, 224

# parent 디렉토리, 불러올 파일들이 있는 디렉토리명, 저장할 곳의 디렉토리 명
def create_brightness_image(data_path, dir_name, save_dir_name):
    train_data_path = os.path.join(data_path, dir_name)      # data_path와 train데이터를 잇는다.
    save_data_path = os.path.join(data_path, save_dir_name)
    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    total = len(images)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        img = Image.open(os.path.join(train_data_path, image_name))
        emphasis_img = emphasis_brightness(img)
        emphasis_img.thumbnail(fix_image_size, Image.ANTIALIAS)
        emphasis_img.save(os.path.join(save_data_path, image_name))
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print('Saving to files image done.')

def create_brightness_gaussian_image(data_path, dir_name, save_dir_name, min=150, max=255):
    train_data_path = os.path.join(data_path, dir_name)      # data_path와 train데이터를 잇는다.
    save_data_path = os.path.join(data_path, save_dir_name)
    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    total = len(images)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        """
        img = Image.open(os.path.join(train_data_path, image_name))
        emphasis_img = emphasis_brightness(img)
        emphasis_img.thumbnail(fix_image_size, Image.ANTIALIAS)
        emphasis_img.save(os.path.join(save_data_path, image_name))
        """
        image = cv2.imread(os.path.join(train_data_path, image_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, min, max, cv2.THRESH_BINARY)[1]
        cv2.imwrite(os.path.join(save_data_path, image_name), thresh)

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print('Saving to files image done.')

def create_empty_image(data_path, dir_name, save_dir_name):
    train_data_path = os.path.join(data_path, dir_name)      # data_path와 train데이터를 잇는다.
    save_data_path = os.path.join(data_path, save_dir_name)
    images = os.listdir(train_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.
    total = len(images)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        img = Image.open(os.path.join(train_data_path, image_name))
        to_empty_picture(img)
        img.save(os.path.join(save_data_path, image_name))
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print('Saving to files image done.')

#input_file_loc = input('input_file_loc: ')
#output_file_loc = input('output_file_loc: ')
#input_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire.jpg"
#output_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire_ycbcr.jpg"
#emphasis_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire_emphasis.jpg"
#test_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire_test.jpg"

#myimage = Image.open(input_file_loc)

#convert_myimage = to_ycbcr_img(myimage)
#convert_myimage.save(output_file_loc)

#emphasis_img = emphasis_brightness(myimage)
#emphasis_img.save(emphasis_file_loc)

#myimage.save(test_file_loc)

if __name__ == '__main__':
    only_fire_dir_name = 'only_fire'                # 불러올 파일들이 있는 디렉토리
    only_fire_save_dir_name = 'only_fire_bright'    # 이미지 변환 후 저장할 디렉토리
    fire_dir_name = 'fire'
    fire_save_dir_name = 'fire_bright'
    nonfire_dir_name = 'nonfire'
    nonfire_save_dir_name = 'nonfire_bright'
    myfire_dir_name = 'myfire'
    myfire_save_dir_name = 'myfire_bright'

    train_data_path = r'data\train'
    test_data_path = r'data\test'
    myfire_data_path = 'data'

    my_empty_dir = 'temp'
    #create_brightness_image(train_data_path, only_fire_dir_name, only_fire_save_dir_name)
    #create_brightness_gaussian_image(train_data_path, only_fire_dir_name, only_fire_save_dir_name, 100, 255)
    #create_brightness_gaussian_image(train_data_path, nonfire_dir_name, nonfire_save_dir_name)
    #create_brightness_gaussian_image(train_data_path, fire_dir_name, fire_save_dir_name)

    #create_brightness_gaussian_image(test_data_path, nonfire_dir_name, nonfire_save_dir_name)
    #create_brightness_gaussian_image(test_data_path, fire_dir_name, fire_save_dir_name)

    #create_nonfire_brightness_data(train_data_path)
    #create_brightness_gaussian_image(myfire_data_path, myfire_dir_name, myfire_save_dir_name, 100, 255)
    create_empty_image(myfire_data_path, my_empty_dir, myfire_save_dir_name)