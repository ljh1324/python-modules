# 이미지 결합

import argparse
import cv2


def handle_channel3():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
                    help = "Path to the image")
    args= vars(ap.parse_args())
    img= cv2.imread(args["image"], 1)
    b, g, r = cv2.split(img)
    mimg = cv2.merge((b, g, r))
    cv2.imshow('frame', mimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    handle_channel3()