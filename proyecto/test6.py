import SimpleITK
import matplotlib.pyplot as plt
import utils
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from scipy import ndimage as ndi
from skimage.color import label2rgb
import matplotlib.patches as mpatches


def remove_noise(sitk_image):
    smooth_filter = SimpleITK.CurvatureFlowImageFilter()
    smooth_filter.SetTimeStep(0.125)
    smooth_filter.SetNumberOfIterations(5)
    img_smooth = smooth_filter.Execute(sitk_image)
    return SimpleITK.GetArrayFromImage(img_smooth)


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


def get_body_mask(image_array):
    body_threshold = threshold_otsu(image_array)
    body_mask = image_array > body_threshold
    return body_mask


def get_body(image_array):
    body_mask = get_body_mask(image_array)
    body = np.multiply(body_mask, image_array)
    body[body < 0] = 0
    return body


def get_bone_mask(image_array):
    bone_mask = np.zeros_like(image_array)
    for z in range(0, image_array.shape[0]):
        bone_threshold = threshold_otsu(image_array[z, :, :])
        bone_mask[z, :, :] = image_array[z, :, :] > bone_threshold
        bone_mask[z, :, :] = remove_small_objects(bone_mask[z, :, :].astype(bool), 200)
        # bone_mask[z, :, :] = ndi.binary_fill_holes(bone_mask[z, :, :].astype(bool))

    return bone_mask


def get_bone(image_array):
    bone_mask = get_bone_mask(image_array)
    bone = np.multiply(bone_mask, image_array)
    bone[bone < 0] = 0
    return bone


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"


reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()

img_original_array = SimpleITK.GetArrayFromImage(img_original)
img_smooth_array = remove_noise(img_original)


legs = get_legs(img_smooth_array)
# legs_orginal = get_legs(img_original_array)

ini = 42
end = 49
for leg_key in legs.keys():
    if leg_key == 'right_leg':
        body_m = get_body(legs['right_leg'])
        for z in range(0, img_original.GetDepth()):
            if ini <= z <= end:
                utils.np_show(body_m[z, :, :])

                bone_threshold = threshold_otsu(body_m[z, :, :])
                bone_mask = body_m[z, :, :] > bone_threshold
                bone_mask = remove_small_objects(bone_mask.astype(bool), 200)
                utils.np_show(bone_mask)

plt.show()




















