import utils
import valley
import segmentation
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import time


def get_sides_center_coordinates(image, remove_small_objects_size=80):
    bone = np.zeros_like(image)
    bone[image < 200] = 0
    bone[image > 200] = 1

    bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
    label_image = label(bone)

    centroids = []
    for region in regionprops(label_image):
        centroids.append(region.centroid)
    return centroids


def get_rows(row):
    row_ini = 0
    if row - 128 > 0:
        row_ini = row - 128
    row_end = row_ini + 256
    return {'row_ini': row_ini, 'row_end': row_end}


def get_legs(image_array):
    centroids = get_sides_center_coordinates(image_array[0, :, :])
    row1 = centroids[0][0].astype(int)
    rows = get_rows(row1)
    right_leg = image_array[:, rows['row_ini']:rows['row_end'], 0:256]

    row2 = centroids[1][0].astype(int)
    rows = get_rows(row2)
    left_leg = image_array[:, rows['row_ini']:rows['row_end'], 256:]

    return {'right_leg': right_leg, 'left_leg': left_leg}


def show_img(img, ini=35, end=55):
    i = 0
    for k in range(ini, end):
        im = img[k, :, :]
        utils.np_show(im)
        i += 1
        if i == 20:
            plt.show()
            i = 0
    plt.show()


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"

t0 = time.time()
img_original_array, img_smooth_array = utils.load_dicom(PathDicom)
legs = get_legs(img_smooth_array)
emphasized_imgs = {}
bone_masks = {}
boundaries = {}

for leg_key in legs.keys():
    if leg_key == 'right_leg':
        leg = legs[leg_key]
        emphasized_imgs[leg_key] = valley.get_valley_emphasized_image(leg)
        bone_masks[leg_key] = segmentation.initial_segmentation(emphasized_imgs[leg_key])
        segmentation.iterative_adaptative_reclassification2(emphasized_imgs[leg_key], bone_masks[leg_key])

        # show_img(bone_masks[leg_key])
        # result = np.zeros_like(emphasized_imgs[leg_key])
        # np.multiply(emphasized_imgs[leg_key], bone_masks[leg_key], result)
        # print len(boundaries[leg_key])
        # for v in boundaries[leg_key]:
        #     result[v[0], v[1], v[2]] = 1
        # show_img(result)
t1 = time.time()
# print t1-t0  # ----- 26.53049016


# centroids = clustering.fuzzy_cmeans(boundaries['right_leg'], emphasized_imgs['right_leg'])
# clustering.modified_fuzzy_cmeans(boundaries['right_leg'], emphasized_imgs['right_leg'])
