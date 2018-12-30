from PIL import Image
import os


fixed_x_size = 224
fixed_y_size = 224


def image_merge(src_image, add_image, x_ratio, y_ratio, threshold):
    src_img = Image.open(src_image)
    fixed_src_img = src_img.resize((fixed_x_size, fixed_y_size))        # 사진 크기 조절

    add_img = Image.open(add_image)
    #resized_add_img = add_img.resize(fixed_x_size * x_ratio, fixed_y_size * y_ratio)
    #add_x_size, add_y_size = resized_add_img.size[0], resized_add_img.size[1]
    add_x_size, add_y_size = add_img.size[0], add_img.size[1]
    add_px_data = add_img.load()                                      # add_img의 pxel data load

    background = add_px_data[0, 0]
    for i in range(add_x_size):
        for j in range(add_y_size):
            p = add_px_data[i, j]
            if abs(p[0] - background[0]) <= threshold and abs(p[1] - background[1]) <= threshold and abs(p[2] - background[2] <= threshold):
                add_px_data[i, j] = (77, 77, 77)    # 특수값으로 초기화

    x_start = int(fixed_x_size * x_ratio)       # 이미지를 복사 붙여넣기할 x 시작점
    y_start = int(fixed_y_size * y_ratio)

    area = (0, 0, fixed_x_size, fixed_y_size)

    new_img = Image.new("RGB", (fixed_x_size, fixed_y_size), (256, 256, 256))
    new_img.paste(fixed_src_img, area)  # src_img를 new_img에 붙여넣음

    new_px_data = new_img.load()
    for i in range(add_x_size):
        for j in range(add_y_size):
            if x_start + i < fixed_x_size and y_start + j < fixed_y_size and add_px_data[i, j] != (77, 77, 77):
                new_px_data[x_start + i, y_start + j] = add_px_data[i, j]

    return new_img


def image_merge2(src_image, add_image, x_start, y_start, threshold):
    src_img = Image.open(src_image)
    fixed_src_img = src_img.resize((fixed_x_size, fixed_y_size))        # 사진 크기 조절

    add_img = Image.open(add_image)
    #resized_add_img = add_img.resize((int(fixed_x_size * 0.5), int(fixed_y_size * 0.5)))
    #add_x_size, add_y_size = resized_add_img.size[0], resized_add_img.size[1]
    add_x_size, add_y_size = add_img.size[0], add_img.size[1]
    add_px_data = add_img.load()                                      # add_img의 pxel data load

    background = add_px_data[0, 0]
    for i in range(add_x_size):
        for j in range(add_y_size):
            p = add_px_data[i, j]
            if abs(p[0] - background[0]) <= threshold and abs(p[1] - background[1]) <= threshold \
                    and abs(p[2] - background[2] <= threshold):
                add_px_data[i, j] = (77, 77, 77)    # 특수값으로 초기화

    area = (0, 0, fixed_x_size, fixed_y_size)

    new_img = Image.new("RGB", (fixed_x_size, fixed_y_size), (256, 256, 256))
    new_img.paste(fixed_src_img, area)  # src_img를 new_img에 붙여넣음

    new_px_data = new_img.load()
    for i in range(add_x_size):
        for j in range(add_y_size):
            if x_start + i < fixed_x_size and y_start + j < fixed_y_size and add_px_data[i, j] != (77, 77, 77):
                new_px_data[x_start + i, y_start + j] = add_px_data[i, j]

    return new_img


def black_white_image_merge(src_image, add_image, x_start, y_start, threshold):
    src_img = Image.open(src_image)
    fixed_src_img = src_img.resize((fixed_x_size, fixed_y_size))        # 사진 크기 조절

    add_img = Image.open(add_image)
    #resized_add_img = add_img.resize((int(fixed_x_size * 0.5), int(fixed_y_size * 0.5)))
    #add_x_size, add_y_size = resized_add_img.size[0], resized_add_img.size[1]
    add_x_size, add_y_size = add_img.size[0], add_img.size[1]
    add_px_data = add_img.load()                                      # add_img의 pxel data load

    background = add_px_data[0, 0]                                   # 흑백 데이터는 하나의 int이다
    for i in range(add_x_size):
        for j in range(add_y_size):
            p = add_px_data[i, j]
            if abs(p - background) <= threshold:                     # 차이가 threshold보다 작을 경우
                add_px_data[i, j] = 77                               # 특수값으로 초기화

    area = (0, 0, fixed_x_size, fixed_y_size)

    new_img = Image.new("RGB", (fixed_x_size, fixed_y_size), (256, 256, 256))
    new_img.paste(fixed_src_img, area)  # src_img를 new_img에 붙여넣음

    new_px_data = new_img.load()
    for i in range(add_x_size):
        for j in range(add_y_size):
            if x_start + i < fixed_x_size and y_start + j < fixed_y_size and add_px_data[i, j] != 77:
                new_px_data[x_start + i, y_start + j] = (256, 256, 256)

    return new_img


def create_merge_image(src_dir, add_dir, want_save_dir):
    src_images = os.listdir(src_dir)
    add_images = os.listdir(add_dir)

    idx = 0
    for src_img in src_images:
        for add_img in add_images:
            for x_ratio in range(3, 6):
                for y_ratio in range(1, 7):
                    new_img = image_merge(os.path.join(src_dir, src_img), os.path.join(add_dir, add_img),
                                          x_ratio, y_ratio, 20)
                    new_img.save(os.path.join(want_save_dir, "result") + str(idx) + '.jpg', 'JPEG')
                    idx += 1


def create_merge_image2(src_dir, add_dir, want_save_dir):
    src_images = os.listdir(src_dir)
    add_images = os.listdir(add_dir)

    print('src images len : ', len(src_images))
    print('add images len : ', len(add_images))

    total_len = len(src_images)
    idx = 0
    file_idx = 0
    cur_image = 0
    for src_img in src_images:
        if cur_image % 100 == 0:
            print('Done: {0}/{1} images'.format(cur_image, total_len))
        cur_image += 1
        for start_x in range(0, 50, 10):
            for start_y in range(0, 50, 10):
                new_img = image_merge2(os.path.join(src_dir, src_img), os.path.join(add_dir, add_images[idx]),
                                       start_x, start_y, 20)
                new_img.save(os.path.join(want_save_dir, "result") + str(file_idx) + '.jpg', 'JPEG')
                idx += 1
                file_idx += 1
                idx %= len(add_images)


def create_merge_black_white_image(src_dir, add_dir, want_save_dir):
    src_images = os.listdir(src_dir)
    add_images = os.listdir(add_dir)

    print('src images len : ', len(src_images))
    print('add images len : ', len(add_images))

    total_len = len(src_images)
    idx = 0
    file_idx = 0
    cur_image = 0
    for src_img in src_images:
        if cur_image % 100 == 0:
            print('Done: {0}/{1} images'.format(cur_image, total_len))
        cur_image += 1
        for start_x in range(0, 50, 10):
            for start_y in range(0, 50, 10):
                new_img = black_white_image_merge(os.path.join(src_dir, src_img),
                                                  os.path.join(add_dir, add_images[idx]), start_x, start_y, 20)
                new_img.save(os.path.join(want_save_dir, "result") + str(file_idx) + '.jpg', 'JPEG')
                idx += 1
                file_idx += 1
                idx %= len(add_images)

view_dir = r'data\unet\view'
onlyfire_dir = r'data\unet\onlyfire'

view_bright_dir = r'data\unet\view_bright'
onlyfire_bright_dir = r'data\unet\onlyfire_bright'

save_dir = r'data\unet\result'
save_mask_dir = r'data\unet\result_mask'

create_merge_image2(view_dir, onlyfire_dir, save_dir)
create_merge_black_white_image(view_bright_dir, onlyfire_bright_dir, save_mask_dir)
