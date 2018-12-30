from __future__ import print_function
import argparse
import cv2


def access_color_pixel1():
    ap= argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
        help = "Path to the image")
    args= vars(ap.parse_args())
    image = cv2.imread(args["image"], 1)
    (b, g, r) = image[0, 0] # px= image[0, 0], b = px[0]
    print("(0, 0) R: {}, G: {}, B: {}".format(r, g, b))
    image[0, 0] = (0, 0, 255) # image[0, 0] = [0, 0, 255]
    (b, g, r) = image[0, 0]
    print("(0, 0) R: {}, G: {}, B: {}".format(r, g, b))

if __name__ == "__main__" :
    access_color_pixel1()