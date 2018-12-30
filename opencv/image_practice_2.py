import argparse
import cv2
from copy import deepcopy


def image_invert_using_pixel():
    ap = argparse.ArgumentParser()
    # -i : 줄임말, --image : argument
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image")
    args = vars(ap.parse_args())
    img = cv2.imread(args["image"], 0)
    dest = deepcopy(img)

    start = cv2.getTickCount()
    dest = 255 - dest
    end = cv2.getTickCount()

    duration = (end - start) / cv2.getTickFrequency()

    cv2.imshow('frame', dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(duration)

if __name__ == '__main__':
    image_invert_using_pixel()