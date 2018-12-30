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

    width = dest.shape[1]     # 열의 길이
    height = dest.shape[0]     # 행의 길이

    start = cv2.getTickCount()
    for x in range(width):
        for y in range(height):
            dest[y, x] = 255 - dest[y, x]
    end = cv2.getTickCount()

    duration = (end - start) / cv2.getTickFrequency()

    cv2.imshow('frame', dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(duration)

if __name__ == '__main__':
    image_invert_using_pixel()