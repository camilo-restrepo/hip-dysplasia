import SimpleITK
import matplotlib.pyplot as plt
import utils
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, convex_hull_object
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import math
from extras import morphsnakes


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
        stats = utils.get_stats_without_background(image_array[z, :, :])
        bone_mask[z, :, :] = image_array[z, :, :] > stats['mean']
        bone_mask[z, :, :] = ndi.binary_fill_holes(bone_mask[z, :, :].astype(bool))
        bone_mask[z, :, :] = remove_small_objects(bone_mask[z, :, :].astype(bool), 200)
        bone_mask[z, :, :] = clear_border(bone_mask[z, :, :])
        bone_mask[z, :, :] = convex_hull_object(bone_mask[z, :, :])

    return bone_mask


def get_bone(image_array):
    bone_mask = get_bone_mask(image_array)
    bone = np.multiply(bone_mask, image_array)
    # bone[bone < 0] = 0
    return bone


PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"


reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()

img_original_array = SimpleITK.GetArrayFromImage(img_original)
img_smooth_array = remove_noise(img_original)


legs = get_legs(img_smooth_array)
# legs_orginal = get_legs(img_original_array)


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u


ini = 42
end = 49
for leg_key in legs.keys():
    if leg_key == 'right_leg':
        body_m = get_body(legs[leg_key])
        bone_m = get_bone_mask(body_m)
        bone_m1 = get_bone(body_m)
        lb_img = label(bone_m)

        for z in range(0, img_original.GetDepth()):
            if ini <= z <= end:
                # utils.np_show(body_m[z, :, :])
                # utils.np_show(bone_m[z, :, :])
                region = regionprops(lb_img[z, :, :])[0]
                minr, minc, maxr, maxc = region.bbox
                dist = math.hypot(maxr - minr, maxc - minc)

                macwe = morphsnakes.MorphACWE(bone_m1[z, :, :], smoothing=0, lambda1=1, lambda2=2)
                macwe.levelset = circle_levelset(bone_m1[z, :, :].shape, (region.centroid[0], region.centroid[1]),
                                                 dist/2)
                e = morphsnakes.evolve_visual(macwe, num_iters=190)
                utils.np_show(e)


plt.show()




