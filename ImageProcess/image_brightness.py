from PIL import Image
import numpy as np
import copy

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
    print(ycbcr_mean)
    y_mean = ycbcr_mean[0]
    cb_mean = ycbcr_mean[1]
    cr_mean = ycbcr_mean[2]
    pix = img_ycbcr.load()
    threshold = 35

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            y = pix[i, j][0]
            cb = pix[i, j][1]
            cr = pix[i, j][2]
            if (y > y_mean and cb > cb_mean and cr > cr_mean and abs(cb - cr) >= threshold):
                pix[i, j] = (255, 255, 255)
            else:
                pix[i, j] = (0, 0, 0)
    return img_ycbcr

#input_file_loc = input('input_file_loc: ')
#output_file_loc = input('output_file_loc: ')
input_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire.jpg"
output_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire_ycbcr.jpg"
emphasis_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire_emphasis.jpg"
test_file_loc = "D:\\MyPythonProject\\ImageProcess\\fire_test.jpg"

myimage = Image.open(input_file_loc)

convert_myimage = to_ycbcr_img(myimage)
convert_myimage.save(output_file_loc)

emphasis_img = emphasis_brightness(myimage)
emphasis_img.save(emphasis_file_loc)

myimage.save(test_file_loc)

