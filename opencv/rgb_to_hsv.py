import cv2
import argparse
from copy import deepcopy


def rgb_to_hsv(img_file):
    img = cv2.imread(img_file, 1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', hsv_img)
    cv2.waitKey(0)


def detect_light_object(img_file):  # 논문 참고: YCbCr칼라 모델에서 화재의 움직임 정보를 이용한 화재검출 알고리즘
    r = 35
    a = 150

    img = cv2.imread(img_file, 1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    width = hsv_img.shape[1]
    height = hsv_img.shape[0]

    for i in range(height):
        for j in range(width):
            y = hsv_img.item(i, j, 0)
            cr = hsv_img.item(i, j, 1)
            cb = hsv_img.item(i, j, 2)
            f1 = 1 if (cr - cb) >= r else 0
            f2 = 1 if y >= a else 0
            result = (f1 * f2) * 255
            hsv_img.itemset((i, j, 0), result)
            hsv_img.itemset((i, j, 1), result)
            hsv_img.itemset((i, j, 2), result)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', hsv_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # -i : 줄임말, --image : argument
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image")
    ap.add_argument("-m", "--mode", required=True,
                    help="Mode 1 = normal, Mode 2 = detected")

    args = vars(ap.parse_args())
    img_file = args["image"]
    mode = args["mode"]

    if int(mode) == 1:
        rgb_to_hsv(img_file)
    elif int(mode) == 2:
        detect_light_object(img_file)
