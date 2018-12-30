import numpy as np
import cv2

def hog(img):
    cell_size = (8, 8)  # height x width in pixels
    block_size = (2, 2)  # hegiht x width in cells, for nomalization, 2 : 2 cell
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    # // : 몫을 구하는 연산자
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),     # window의 크기
                            _blockSize=(block_size[1] * cell_size[1],                   # block의 총 크기
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),                  # window의 stride(얼만큼 이동시킬 것인가?)
                            _cellSize=(cell_size[1], cell_size[0]),                     # cell size
                            _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])      # 총 cell의 크기 지정
    hog_feats = hog.compute(img)\
                   .reshape(n_cells[1] - block_size[1] + 1,                     # block을 overrapping하며 덧대면서 붙임으로 hog의 각 셀의 결과는 n_cells - block_size + 1이 된다.
                            n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1],                        # 각 블럭이 가지는 크기
                            nbins) \
                   .transpose((1, 0, 2, 3, 4))  # index blocks by rows first

    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))                      # 각 cell마다 nbins의 방향 히스토그램을 가지고 있다.

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)            # n_cells[0] x n_cells[1] x 1 배열을 0으로 채운다.

    # gradients의 크기 : n_cells[0] x n_cells[1] x nbins
    # overrapping 된 블락을 펼친다.
    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                      off_x:n_cells[1] - block_size[1] + off_x + 1] += hog_feats[:, :, off_y, off_x, :] # off_y 에서 n_cells[0] - block_size[0] + off_y + 1 까지 hog_feats를 더해준다.
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                       off_x:n_cells[1] - block_size[1] + off_x + 1] += 1                               # count도 마찬가지로 더해준다.

    # Average gradients
    gradients /= cell_count
    return gradients