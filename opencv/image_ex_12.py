# 이미지 분리

import argparse
import cv2


def handle_channel12():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
                    help = "Path to the image")
    args= vars(ap.parse_args())
    img = cv2.imread(args["image"], 1)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    cv2.imshow('framer', r)
    cv2.imshow('frameg', g)
    cv2.imshow('frameb', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__" :
    handle_channel2()