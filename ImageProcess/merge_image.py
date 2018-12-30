import glob
from PIL import Image

def calculateSize(files):
    size_x = []
    size_y = []

    for file in files:
        image = Image.open(file)
        size_x.append(image.size[0])
        size_y.append(image.size[1])
    #print(size_x)
    #print(size_y)

    x_min = min(size_x)
    y_min = min(size_y)
    total_x_size = x_min * len(files)   # 다른 이미지를 붙일 것이기에 가로로는 크기를 파일 갯수만큼 조절해 주어야 합니다

    #print("x_min: ", x_min)
    #print("y_min: ", y_min)
    #print("total_x_size: ", total_x_size)

    return x_min, y_min, total_x_size

def resizeToMin(files, x_min, y_min, x_size):
    file_list = []
    for file in files:
        image = Image.open(file)
        resized_file = image.resize((x_min, y_min))
        file_list.append(resized_file)
        print(resized_file.size)
        resized_file.show()
    return file_list, x_size, x_min, y_min

def imageMerge(img_list, x_size, x_min, y_min):
    new_image = Image.new("RGB", (x_size, y_min), (256, 256, 256))  # WHITE Canverse
    print("X_size: ", x_size)
    print(len(img_list))

    for index in range(len(img_list)):
        area = ((index * x_min), 0, (x_min * (index + 1)), y_min)
        new_image.paste(img_list[index], area)
    new_image.show()
    return new_image

def imageMerge2(originImageFile, addImageFile, x_ratio, y_ratio):
    orgImg = Image.open(originImageFile)
    addImg = Image.open(addImageFile)

    org_x_size, org_y_size = orgImg.size[0], orgImg.size[1]
    print(org_x_size)
    print(org_y_size)

    add_x_size, add_y_size = addImg.size[0], addImg.size[1]
    print(add_x_size)
    print(add_y_size)

    add_pxData = addImg.load()
    background = add_pxData[0, 0]
    for i in range(add_x_size):
        for j in range(add_y_size):
            p = add_pxData[i, j]
            if (abs(p[0] - background[0]) <= 50 and abs(p[1] - background[1]) <= 50 and abs(p[2] - background[2] <= 50)):
                add_pxData[i, j] = (256, 256, 256)
    x_start = int(org_x_size * x_ratio)
    y_start = int(org_y_size * y_ratio)

    area1 = (0, 0, org_x_size, org_y_size)
    area2 = (x_start, y_start, x_start + add_x_size, y_start + add_y_size)
    newImg = Image.new("RGB", (org_x_size, org_y_size), (256, 256, 256))
    newImg.paste(orgImg, area1)
    newImg.paste(addImg, area2)

    return newImg

def imageMerge3(originImageFile, addImageFile, x_ratio, y_ratio):
    orgImg = Image.open(originImageFile)
    addImg = Image.open(addImageFile)

    org_x_size, org_y_size = orgImg.size[0], orgImg.size[1]
    print(org_x_size)
    print(org_y_size)

    add_x_size, add_y_size = addImg.size[0], addImg.size[1]
    print(add_x_size)
    print(add_y_size)

    add_pxData = addImg.load()
    background = add_pxData[0, 0]
    for i in range(add_x_size):
        for j in range(add_y_size):
            p = add_pxData[i, j]
            if (abs(p[0] - background[0]) <= 50 and abs(p[1] - background[1]) <= 50 and abs(p[2] - background[2] <= 50)):
                add_pxData[i, j] = (77, 77, 77)

    x_start = int(org_x_size * x_ratio)
    y_start = int(org_y_size * y_ratio)

    area1 = (0, 0, org_x_size, org_y_size)

    newImg = Image.new("RGB", (org_x_size, org_y_size), (256, 256, 256))
    newImg.paste(orgImg, area1)

    new_pxData = newImg.load()
    for i in range(add_x_size):
        for j in range(add_y_size):
            if (x_start + i < org_x_size and y_start + j < org_y_size and add_pxData[i, j] != (77, 77, 77)):
                new_pxData[x_start + i, y_start + j] = add_pxData[i, j]
    return newImg

target_dir = "D:\\MyPythonProject\\ImageProcess\\image\\"
#files = glob.glob(target_dir + "*.*")

#print(len(files))
#print(files)

#x_min, y_min, x_size = calculateSize(files)
#img_list, x_size, x_min, y_min = resizeToMin(files, x_min, y_min, x_size)
#new_image = imageMerge(img_list, x_size, x_min, y_min)
#new_image.save(target_dir + "result.png", "PNG")

streetImg = target_dir + "view.jpg"
fireImg = target_dir + "fire.jpg"
newImg = imageMerge2(streetImg, fireImg, 0.2, 0.2)
newImg.save(target_dir + "result2.png", "PNG")
newImg2 = imageMerge3(streetImg, fireImg, 0.2, 0.2)
newImg2.save(target_dir + "result3.png", "PNG")

fireFiles = glob.glob(target_dir + "fire*.*")
streetFiles = glob.glob(target_dir + "view*.*")

idx = 0
for y_ratio in range(3, 6):
    for x_radio in range(1, 7):
        for fire in fireFiles:
            for street in streetFiles:
                newImg = imageMerge3(street, fire, x_radio/10, y_ratio/10)
                newImg.save(target_dir + "result" + str(idx) + ".png", "PNG")
                idx = idx + 1
