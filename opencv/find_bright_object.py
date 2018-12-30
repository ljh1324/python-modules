import cv2

# http://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/

# load the image, convert it to grayscale, and blur it
image = cv2.imread('fire.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('image', thresh)
cv2.imwrite('fire_img.jpg', thresh)
cv2.waitKey(0)
