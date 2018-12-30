# pip install pillow

from PIL import Image
import numpy as np
import colorsys

img = Image.open("D:\\MyPythonProject\\ImageProcess\\image\\result20.png")

def make_int_tuple(tuple):
    return (int(tuple[0]), int(tuple[1]), int(tuple[2]))
px = img.load()
print(px)
pix = np.asarray(img)
print(pix.shape)
for i in range(img.size[0]):
    for j in range(img.size[1]):
        hsv = colorsys.rgb_to_hsv(px[i, j][0], px[i, j][1], px[i, j][2])
        px[i, j] = make_int_tuple(hsv)
    print()


img.save("D:\\MyPythonProject\\ImageProcess\\fire_with_street_to_hsv.jpg")
print(img.size)
print(img.format)
