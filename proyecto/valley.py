import numpy as np
from skimage.morphology import closing


def get_valley_image(image):
    selem = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
    valley_img = closing(image, selem)
    valley_img = valley_img - image
    return valley_img


def get_valley_emphasized_image(image):
    valley_image = get_valley_image(image)
    valley_image = image - valley_image
    return valley_image
