# 이미지 속성 확인하기

from __future__ import print_function
import argparse
import cv2


def info_image():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
                    help = "Path to the image")
    args= vars(ap.parse_args())
    img = cv2.imread(args["image"])
    print("width: {} pixels".format(img.shape[1]))      # 열의 길이
    print("height: {} pixels".format(img.shape[0]))     # 행의 길이
    print("channels: {}".format(img.shape[2]))
    print(img.size)
    print(img.dtype)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    info_image()