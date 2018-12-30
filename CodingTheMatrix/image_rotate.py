# 참고 Coding The Matrix - p.56
# Task 2.4.20

from PIL import Image
import plotting
import cmath

target_dir = "D:\\MyPythonProject\\CodingTheMatrix\\image\\"
frogImgLocation = target_dir + "tt.png"

image = Image.open(frogImgLocation)
pxList = image.load()
x_size = image.size[0]
y_size = image.size[1]

background = pxList[0, 0]
dotList = []
for i in range(x_size):
    for j in range(y_size):
        if (pxList[i, j] != background):
            dotList.append(i + (y_size - j) * 1j)
print(dotList)

delta = cmath.pi / 4
dotList = [(dot * (cmath.e**(delta * 1j))) / 2 for dot in dotList]

plotting.plot(dotList, 'ro--', 120)