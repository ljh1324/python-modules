# gray 픽셀 접근

from __future__ import print_function
import argparse
import cv2


def access_gray_pixel1():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"], 0)        # load gray scale, 8bit, 0(검은색) ~ 255(흰색)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', image)
    cv2.waitKey(0)

    value = image[0, 0]
    print("(0, 0) value: {}".format(value))
    image[0, 0] = 255
    value = image[0, 0]
    print("(0, 0) value: {}".format(value))

if __name__ == "__main__" :
    access_gray_pixel1()
