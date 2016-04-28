import numpy as np


def pixel_belongs_to_boundary(img_array, x, y, z):
    pixel = img_array[z, x, y]
    if pixel != 0:
        neighbors = [
            img_array[z, x - 1, y - 1],
            img_array[z, x - 1, y],
            img_array[z, x - 1, y + 1],
            img_array[z, x, y - 1],
            img_array[z, x, y + 1],
            img_array[z, x + 1, y - 1],
            img_array[z, x + 1, y],
            img_array[z, x + 1, y + 1],
            img_array[z - 1, x, y],
            img_array[z + 1, x, y]
        ]

        for n in neighbors:
            if n == 0:
                return True
    return False


def compute_boundary(image):
    boundaries_array = np.zeros_like(image)
    width = image.shape[1]
    height = image.shape[2]
    depth = image.shape[0]
    e_b_temp = {}

    for z in range(0, depth):
        e_b_list = []
        for index, x in np.ndenumerate(image[z, :, :]):
            if x != 0:
                if 0 < index[0] < height-1 and 0 < index[1] < width-1 and 0 < z < depth-1:
                    if pixel_belongs_to_boundary(image, index[0], index[1], z):
                        boundaries_array[z, index[0], index[1]] = x
                        e_b_list.append(index)
                e_b_temp[z] = e_b_list

    e_b = set()
    for k in e_b_temp.keys():
        point_list = e_b_temp[k]
        for p in point_list:
            point = (k, p[0], p[1])
            e_b.add(point)

    return boundaries_array, e_b
