import argparse
import cv2


def handle_roi():
    ap= argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
                    help = "Path to the image")
    args= vars(ap.parse_args())
    img = cv2.imread(args["image"])
    corner = img[0:100, 0:100]
    cv2.imshow("Corner", corner)
    img[0:100, 0:100] = (0, 255, 0)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    handle_roi()