"""
from PIL import Image
import numpy as np
import colorsys

input_link = input("link : ")
img = Image.open(input_link)


def make_color_tuple(color):
    return (color, color, color)

def labeling(px, x, y, color):
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    if (x < 0 or y < 0)
        return

    if (px[x, y][0] == 0 and px[x, y][1] == 0 and px[x, y][2] == 0):
        px[x, y] = make_color_tuple(color)
        for i in range(len(dx)):

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
"""

"""
from PIL import Image
import numpy as np
import colorsys
import sys

input_link = input("link : ")
img = Image.open(input_link)


def make_color_tuple(color):
    return (color, color, color)


def labeling(px, x, y, color, width, height):
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    if (x < 0 or y < 0 or x >= width or y >= height):
        return

    print (x, y)
    if (px[y, x][0] == 0 and px[y, x][1] == 0 and px[y, x][2] == 0):
        px[y, x] = make_color_tuple(color)
        for i in range(len(dx)):
            labeling(px, x + dx[i], y + dy[i], color, width, height)


def all_labeling(px, width, height):
    color = 10
    for i in range(width):
        for j in range(height):
            if (px[j, i] == (0, 0, 0)):
                labeling(px, i, j, color, width, height)
                color += 30
px = img.load()
print(img.size)
print(px)

sys.setrecursionlimit(10000)
all_labeling(px, img.size[1], img.size[0])
img.save("D:\\MyPythonProject\\ImageProcess\\tt.jpg")
"""
from PIL import Image
import numpy as np
import colorsys
import sys
import random

def make_color_tuple():
    return (random.randrange(0,255), random.randrange(0,255), random.randrange(0,255))

def init_matrix(width, height):
    matrix = [[0 for col in range(width)] for row in range(height)]
    #print(matrix)
    return matrix

def all_labeling(px, width, height):
    color = 10
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]

    for i in range(width):
        for j in range(height):
            x_loc = i
            y_loc = j
            if (px[y_loc, x_loc][0] >= 200 and px[y_loc, x_loc][1] >= 200 and px[y_loc, x_loc][2] >= 200):
                stack = [(x_loc, y_loc)]    # Stack에 (x_loc, y_loc)을 넣어 초기화
                color = make_color_tuple()
                check_list = init_matrix(width, height)

                while (len(stack) != 0):   # Stack이 비어있지 않을 동안 반복
                    print(len(stack))
                    x_loc, y_loc = stack.pop()  # Stack에서 값을 꺼냄
                    print(x_loc, y_loc)
                    if (check_list[y_loc][x_loc] == True):          # 방문된 곳이면 넘어감
                        continue
                    check_list[y_loc][x_loc] = True
                    if (px[y_loc, x_loc][0] >= 200 and px[y_loc, x_loc][1] >= 200 and px[y_loc, x_loc][2] >= 200):

                        px[y_loc, x_loc] = color
                        for i in range(len(dx)):
                            if (x_loc + dx[i] < 0 or y_loc + dy[i] < 0 or x_loc + dx[i] >= width or y_loc + dy[i] >= height):
                                continue
                            else:
                                stack.append((x_loc + dx[i], y_loc + dy[i]))

"""
input_dir_link = input("input dir_link : ")
input_file_name = input("input file_name : ")
input_file_type = input(" input file_type : ")
input_size = int(input("input size: "))

output_dir_link = input("output dir_link : ")
output_file_name = input("output file_name : ")
"""

input_dir_link = "C:\\Users\\이정환\Desktop\\img_data"
input_file_name = "compose_fire"
input_file_type = "jpg"
input_size = 15

output_dir_link = "D:\\MyPythonProject\\ImageProcess"
output_file_name = "labeling_compose_fire"
for i in range(input_size):
    concat_input_link = input_dir_link + "\\" + input_file_name + str(i + 1) + "." + input_file_type
    concat_output_link = output_dir_link + "\\" + output_file_name + str(i) + ".jpg"
    img = Image.open(concat_input_link)
    px = img.load()
    all_labeling(px, img.size[1], img.size[0])
    img.save(concat_output_link)
    print(i)