# 이미지 크기 변경

import argparse
import cv2


def resize_image():
    ap= argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image")
    args= vars(ap.parse_args())
    img = cv2.imread(args["image"])
    r = 400.0 / img.shape[1]        # width를 200으로 변경했을 경우 비율을 저장
    dim = (400, int(img.shape[0] * r))  # height를 변경된 비율만큼 resize

    # perform the actual resizing of the image and show it
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('frame', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    resize_image()
