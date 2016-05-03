import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, diamond
from scipy import ndimage as ndi
import bone_boundary
from extras import cmeans2
import utils
import matplotlib.pyplot as plt
import time


idx = 'idx'
yx = 'yx'
u = 'u'
c = 'c'
alpha = 0.85
p = 2
ncentroids = 2


def initial_segmentation(image):
    result = np.zeros_like(image)
    threshold = threshold_otsu(image)
    binary_img = image > threshold
    np.multiply(image, binary_img, result)
    threshold = threshold_otsu(result)
    binary_img = result > threshold

    selem = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])

    binary_img = closing(binary_img, selem)
    binary_img = ndi.binary_fill_holes(binary_img, structure=selem)

    for z in range(0, image.shape[0]):
        img = binary_img[z, :, :].astype(bool)
        img = ndi.binary_fill_holes(img)
        binary_img[z, :, :] = remove_small_objects(img, 80)
    return binary_img


def get_window(point, image, size=5, depth=2):
    z = point[0]
    x = point[1]
    y = point[2]

    x_ini = x - size
    x_end = x + (size+1)
    y_ini = y - size
    y_end = y + (size+1)
    z_ini = z - depth
    z_end = z + (depth+1)

    window = image[z_ini:z_end, x_ini:x_end, y_ini:y_end]
    return window


def get_data(point, image):
    vecinos_x = get_window(point, image)
    size = vecinos_x.shape[0] * vecinos_x.shape[1] * vecinos_x.shape[2]
    data = np.zeros(size, dtype={'names': [idx, yx, u, c], 'formats': ['3int8', 'f4', 'f4', 'i4']})

    i = 0
    for z in range(0, vecinos_x.shape[0]):
        for x in range(0, vecinos_x.shape[1]):
            for y in range(0, vecinos_x.shape[2]):
                data[i] = ([z, x, y], vecinos_x[z, x, y], 0, 0)
                i += 1
    return data


def replace_volume(point, image, replace, size=5, depth=2):
    result = image.copy()

    z = point[0]
    x = point[1]
    y = point[2]

    x_ini = x - size
    x_end = x + (size+1)
    y_ini = y - size
    y_end = y + (size+1)
    z_ini = z - depth
    z_end = z + (depth+1)

    result[z_ini:z_end, x_ini:x_end, y_ini:y_end] = replace
    return result


def iterative_adaptative_reclassification(image, bone_mask):
    boundaries = bone_boundary.compute_boundary(bone_mask)
    boundaries_old = np.zeros_like(boundaries)
    bone_mask_old = bone_mask.copy()
    t0 = time.time()
    for px in boundaries:
        if 2 < px[0] < image.shape[0]-3 and 11 < px[1] < image.shape[1]-11 and 11 < px[2] < image.shape[2]-11:
            window = get_window(px, image)
            cntr, u_, u0, d, jm, h, fpc = cmeans2.cmeans(np.reshape(window, (1, window.size)), ncentroids, p, 0.001, 1000)
            v = 0
            if cntr[0, 0] < cntr[1, 0]:
                v = 1
            result = np.reshape(u_[v, :], window.shape)

            th = threshold_otsu(result)
            result2 = result > th
            bone_mask = replace_volume(px, bone_mask, result2)
    t1 = time.time()
    print 'total: ', t1-t0
    # selem = np.array([
    #     [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    #     [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    #     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # ])
    # bone_mask = closing(bone_mask, selem)

    for z in range(0, bone_mask.shape[0]):
        bone_mask[z, :, :] = ndi.binary_closing(bone_mask[z, :, :], structure=diamond(3))
        bone_mask[z, :, :] = ndi.binary_fill_holes(bone_mask[z, :, :], structure=diamond(3))

    i = 0
    for k in range(35, 55):
        im = bone_mask[k, :, :]
        utils.np_show(im)
        utils.np_show(bone_mask_old[k, :, :])
        i += 1
        if i == 10:
            plt.show()
            i = 0
    plt.show()

