import os
import glob
from PIL import Image

target_dir = 'C:\\Users\\이정환\\Desktop\\툴즈\\street\\'
files = glob.glob(target_dir + "*.*")

idx = 1
for file in files:
    newImg = Image.open(file).convert('RGB')
    newImg.save(target_dir + "test_street" + str(idx) + ".jpg", "JPEG")
    idx = idx + 1