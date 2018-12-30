import cv2


def load_image1():
    img_file= 'frog.jpg'
    img= cv2.imread(img_file, cv2.IMREAD_COLOR)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__" :
    load_image1()