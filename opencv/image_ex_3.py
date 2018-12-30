import argparse
import cv2


def load_image3():
    ap = argparse.ArgumentParser()
    # -i : 줄임말, --image : argument
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image")
    args= vars(ap.parse_args())
    img = cv2.imread(args["image"])
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    load_image3()