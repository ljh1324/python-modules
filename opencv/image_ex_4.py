import argparse
import cv2


def load_and_save_image():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
                    help = "Path to the image")
    args= vars(ap.parse_args())
    img = cv2.imread(args["image"])
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', img)
    pk = cv2.waitKey(0) & 0xFF  # in 64bit machine, 눌러진 키를 저장
    if pk == ord('c'):  # chr(99)는문자'c'반환
        cv2.imwrite('copy.jpg', img)
        cv2.destroyAllWindows()

if __name__ == "__main__" :
    load_and_save_image()
