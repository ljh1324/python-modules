import numpy as np


def file2csv(input_file, save_file):
    arr = np.load(input_file)

    f_write = open(save_file, 'w')
    print(arr.shape)
    if len(arr.shape) == 2:
        numpy_arr = arr[..., np.newaxis]

    for k in range(arr.shape[2]):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1] - 1):
                f_write.write(str(arr[i, j, k]) + ',')
            f_write.write(str(arr[i, j, k]) + '\n')


def numpy2csv(numpy_arr, save_file):
    f_write = open(save_file, 'w')
    print(numpy_arr.shape)
    if len(numpy_arr.shape) == 2:
        numpy_arr = numpy_arr[..., np.newaxis]

    for k in range(numpy_arr.shape[2]):
        for i in range(numpy_arr.shape[0]):
            for j in range(numpy_arr.shape[1] - 1):
                f_write.write(str(numpy_arr[i, j, k]) + ',')
            f_write.write(str(numpy_arr[i, j, k]) + '\n')


if __name__ == '__main__':
    a = np.zeros((3, 3))
    numpy2csv(a, 'test2.csv')


