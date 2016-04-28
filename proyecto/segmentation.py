import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing
from scipy import ndimage as ndi


def initial_segmentation(image):
    result = np.zeros_like(image)
    threshold = threshold_otsu(image)
    binary_img = image > threshold
    np.multiply(image, binary_img, result)
    threshold = threshold_otsu(result)
    binary_img = result > threshold
    binary_img[:, 450:, :] = 0

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
