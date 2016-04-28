import utils
import valley
import segmentation
import bone_boundary
import clustering
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


def iterative_adaptative_reclassification(image):

    i = 0
    for z in range(0, end):
        if ini <= z <= end:
            utils.np_show(emphasized_img[z, :, :])
            utils.np_show(segmented_img[z, :, :])
            i += 1
            if i == 10:
                plt.show()
                i = 0
    return result['boundaries_array']


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"

t0 = time.time()
img_original_array, img_smooth_array = utils.load_dicom(PathDicom)
legs = get_legs(img_smooth_array)
emphasized_imgs = {}
bone_masks = {}
boundaries_array = {}
boundaries = {}

for leg_key in legs.keys():
    if leg_key == 'right_leg':
        leg = legs[leg_key]
        emphasized_img = valley.get_valley_emphasized_image(leg)
        emphasized_imgs[leg_key] = emphasized_img
        bone_mask = segmentation.initial_segmentation(emphasized_img)
        bone_masks[leg_key] = bone_mask
        boundary_array, e_b = bone_boundary.compute_boundary(bone_mask)
        boundaries_array[leg_key] = boundary_array
        boundaries[leg_key] = e_b
t1 = time.time()
# print t1-t0 ----- 33.3400089741

# result = boundaries['right_leg']
# result = np.zeros_like(legs['right_leg'])
# np.multiply(emphasized_imgs['right_leg'], bone_masks['right_leg'], result)

centroids = clustering.fuzzy_cmeans(boundaries['right_leg'], emphasized_imgs['right_leg'])

# ini = 20
# end = 50
# i = 0
# for k in range(ini, end):
#     im = result[k, :, :]
#     utils.np_show(im)
#     im[im < 0] = 0
#     utils.show_hist(im)
#     i += 1
#     if i == 10:
#         plt.show()
#         i = 0
#
# plt.show()
