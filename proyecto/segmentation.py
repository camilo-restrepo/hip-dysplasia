import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, diamond
from scipy import ndimage as ndi
import bone_boundary
from extras import cmeans2
import utils
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

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
        binary_img[z, 240:, :] = 0
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

    # bone_mask = closing(bone_mask)
    #
    # for z in range(0, bone_mask.shape[0]):
    #     bone_mask[z, :, :] = closing(bone_mask[z, :, :])
    #     bone_mask[z, :, :] = ndi.binary_fill_holes(bone_mask[z, :, :])
    #
    # i = 0
    # for k in range(35, 55):
    #     im = bone_mask[k, :, :]
    #     utils.np_show(im)
    #     utils.np_show(bone_mask_old[k, :, :])
    #     i += 1
    #     if i == 10:
    #         plt.show()
    #         i = 0
    # plt.show()


def process_px(image, px):
    if 2 < px[0] < image.shape[0]-3 and 11 < px[1] < image.shape[1]-11 and 11 < px[2] < image.shape[2]-11:
        window = get_window(px, image)
        cntr, u, u0, d, jm, h, fpc = cmeans2.cmeans(np.reshape(window, (1, window.size)), ncentroids, p, 0.001, 1000)
        v = 0
        if cntr[0, 0] < cntr[1, 0]:
            v = 1
        result = np.reshape(u[v, :], window.shape)
        th = threshold_otsu(result)
        result = result > th
        return result


def iterative_adaptative_reclassification2(image, bone_mask):
    boundaries = bone_boundary.compute_boundary(bone_mask)
    boundaries_old = np.zeros_like(boundaries)
    bone_mask_old = bone_mask.copy()

    pool = Pool()
    i = 0

    while not np.all(boundaries == boundaries_old):
        func = partial(process_px, image)
        results = pool.map(func, boundaries)

        for idx, px in enumerate(boundaries):
            bone_mask = replace_volume(px, bone_mask, results[idx])

        for z in range(0, bone_mask.shape[0]):
            bone_mask[z, :, :] = closing(bone_mask[z, :, :])
            bone_mask[z, :, :] = ndi.binary_fill_holes(bone_mask[z, :, :])

        boundaries_old = boundaries.copy()
        boundaries = bone_boundary.compute_boundary(bone_mask)
        i += 1
        print i

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
