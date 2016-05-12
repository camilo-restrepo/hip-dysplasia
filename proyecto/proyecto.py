import SimpleITK
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import remove_small_objects, closing, disk, erosion, dilation, square
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.color import label2rgb
from scipy.spatial.distance import euclidean
from skimage.segmentation import clear_border
import region_growing


PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"

RIGHT_LEG = 'right_leg'
LEFT_LEG = 'left_leg'

# --------------------------------------------------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------------------------------------------------


def load_dicom():
    reader = SimpleITK.ImageSeriesReader()
    filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
    reader.SetFileNames(filenames_dicom)
    img_original = reader.Execute()
    original_array = SimpleITK.GetArrayFromImage(img_original)
    return original_array


# --------------------------------------------------------------------------------------------------------------------
# SEPARATE LEGS
# --------------------------------------------------------------------------------------------------------------------


def get_sides_center_coordinates(remove_small_objects_size=80):
    bone = np.zeros_like(img_original_array[0, :, :])
    bone[img_original_array[0, :, :] < 200] = 0
    bone[img_original_array[0, :, :] > 200] = 1

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


def get_legs():
    centroids = get_sides_center_coordinates()
    row1 = centroids[0][0].astype(int)
    rows = get_rows(row1)
    # right_leg = img_smooth_array[:, rows['row_ini']:rows['row_end'], 0:256]
    right_leg = img_original_array[:, rows['row_ini']:rows['row_end'], 0:256]

    row2 = centroids[1][0].astype(int)
    rows = get_rows(row2)
    # left_leg = img_smooth_array[:, rows['row_ini']:rows['row_end'], 256:]
    left_leg = img_original_array[:, rows['row_ini']:rows['row_end'], 256:]

    return {RIGHT_LEG: right_leg, LEFT_LEG: left_leg}


# --------------------------------------------------------------------------------------------------------------------
# REMOVE NOISE
# --------------------------------------------------------------------------------------------------------------------


def remove_noise_curvature_flow():
    sitk_image = SimpleITK.GetImageFromArray(legs[leg_key])
    smooth_filter = SimpleITK.CurvatureFlowImageFilter()
    smooth_filter.SetTimeStep(0.125)
    smooth_filter.SetNumberOfIterations(5)
    img_smooth = smooth_filter.Execute(sitk_image)
    return SimpleITK.GetArrayFromImage(img_smooth)


def remove_noise_anisotropic():
    sitk_image = SimpleITK.GetImageFromArray(legs[leg_key])
    sitk_image = SimpleITK.Cast(sitk_image, SimpleITK.sitkFloat64)

    extract_filter = SimpleITK.ExtractImageFilter()
    extract_filter.SetSize([legs[leg_key].shape[1], legs[leg_key].shape[2], 0])

    smooth_filter = SimpleITK.GradientAnisotropicDiffusionImageFilter()
    smooth_filter.SetTimeStep(0.06)
    smooth_filter.SetNumberOfIterations(50)
    smooth_filter.SetConductanceParameter(0.5)

    img_smooth = SimpleITK.GetArrayFromImage(smooth_filter.Execute(sitk_image))

    # img_smooth = np.zeros(legs[leg_key].shape)
    # for z in range(0, img_smooth.shape[0]):
    #     extract_filter.SetIndex([0, 0, z])
    #     img_smooth[z, :, :] = SimpleITK.GetArrayFromImage(smooth_filter.Execute(sitk_image[:, :, z]))
    return img_smooth


# --------------------------------------------------------------------------------------------------------------------
# VALLEY COMPUTATION
# --------------------------------------------------------------------------------------------------------------------


def get_valley_image():
    image = no_noise[leg_key]
    valley_img = np.zeros_like(image)
    for z in range(0, image.shape[0]):
        valley_img[z, :, :] = closing(image[z, :, :], disk(5))
    valley_img = valley_img - image

    return valley_img


def get_valley_image2():
    image = no_noise[leg_key]
    valley_img = np.zeros_like(image)
    for z in range(0, image.shape[0]):
        valley_img[z, :, :] = closing(image[z, :, :], disk(5))
    valley_img = valley_img - image

    tmp = np.zeros(valley_img.shape)
    tmp[valley_img > 60] = 1
    tmp[valley_img < 60] = 0
    for z in range(0, image.shape[0]):
        tmp[z, :, :] = remove_small_objects(tmp[z, :, :].astype(bool), 20)
        tmp[z, :, :] = ndi.binary_fill_holes(tmp[z, :, :])

    return tmp


def get_valley_emphasized_image():
    image = no_noise[leg_key]
    valley_image = get_valley_image()
    valley_image = image - valley_image
    return valley_image


# --------------------------------------------------------------------------------------------------------------------
# INITIAL SEGMENTATION
# --------------------------------------------------------------------------------------------------------------------


def initial_segmentation():
    valley = get_valley_image2()
    image = emphasized[leg_key]
    binary_img = image > 200
    for z in range(0, image.shape[0]):
        img = binary_img[z, :, :]
        img = clear_border(img)
        img = remove_small_objects(img.astype(bool), 50)
        img = ndi.binary_fill_holes(img)
        # img = closing(img)
        # img = ndi.binary_fill_holes(img)
        binary_img[z, :, :] = img

    binary_img = binary_img - valley
    binary_img[binary_img == -1] = 0

    for z in range(0, image.shape[0]):
        img = binary_img[z, :, :]
        img = ndi.binary_fill_holes(img)
        binary_img[z, :, :] = img
    return binary_img


def initial_segmentation_erosion_dilation():
    binary_img = initial_segmentation()

    selem = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ])

    for z in range(0, binary_img.shape[0]):
        img = binary_img[z, :, :]
        for j in range(0, 2):
            img = erosion(img, selem)

        for j in range(0, 2):
            img = dilation(img, selem)
        binary_img[z, :, :] = img

    return binary_img


def femur_separation():
    image = initial_segmentation_erosion_dilation().copy()
    # init = image.copy()
    coordinates = []
    regions = set()

    if leg_key == LEFT_LEG:
        for z in range(0, image.shape[0]):
            image[z, :, :] = np.fliplr(image[z, :, :])

    i = 0
    while len(coordinates) == 0:
        label_image = label(image[i, :, :])
        for region in regionprops(label_image):
            coordinates.append(region.centroid[1])
        i += 1

    for z in range(i, image.shape[0]):
        label_image = label(image[z, :, :])
        for region in regionprops(label_image):
            centroid = region.centroid[1]
            promedio = np.mean(coordinates)
            distance = euclidean(centroid, promedio)

            # print centroid, promedio, distance
            if centroid < promedio:
                coordinates.append(centroid)
            elif distance < 40:
                coordinates.append(centroid)
            else:
                for c in region.coords:
                    label_image[c[0], c[1]] = 0
                    image[z, c[0], c[1]] = 0

            # print np.mean(coordinates)

        # print'----------'
        if leg_key == LEFT_LEG:
            image[z, :, :] = np.fliplr(image[z, :, :])
            # label_image = np.fliplr(label_image)

        # image_label_overlay = label2rgb(label_image, image=init[z, :, :])

        segmented_leg = segmented[leg_key][z, :, :].copy()
        # segmented_leg = erosion(segmented_leg, square(1))
        # segmented_leg = dilation(segmented_leg, square(1))

        # if 30 <= z <= 51:
        #     fig = plt.figure()
        #     a = fig.add_subplot(1, 4, 1)
        #     imgplot = plt.imshow(segmented_leg, cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 2)
        #     imgplot = plt.imshow(init[z, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 3)
        #     imgplot = plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 4)
        #     imgplot = plt.imshow(segmented_leg + image[z, :, :], cmap='Greys_r', interpolation="nearest")
        #     plt.show()

        seed_points = set()
        for x in range(0, segmented_leg.shape[0]):
            for y in range(0, segmented_leg.shape[1]):
                if image[z, x, y]:
                    seed_points.add((z, x, y))

        region = region_growing.simple_2d_binary_region_growing(segmented_leg, seed_points)
        regions |= region

    result = np.zeros_like(image)
    for px in regions:
        result[px[0], px[1], px[2]] = 1

    selem = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ])

    for z in range(0, result.shape[0]):
        img = result[z, :, :]
        for j in range(0, 3):
            img = dilation(img, selem)
        for j in range(0, 3):
            img = erosion(img, selem)
        result[z, :, :] = img

    return result


# --------------------------------------------------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------------------------------------------------


img_original_array = load_dicom()
legs = get_legs()
no_noise = {}
valleys = {}
emphasized = {}
segmented = {}


for leg_key in legs.keys():
    if leg_key == RIGHT_LEG:
        no_noise[leg_key] = remove_noise_curvature_flow()
        emphasized[leg_key] = get_valley_emphasized_image()
        segmented[leg_key] = initial_segmentation()
        femur = femur_separation()
        # valley = get_valley_image()
        # valley2 = get_valley_image2()

        #         for k in range(30, 51):
        # for k in range(0, no_noise[leg_key].shape[0]):
        #     fig = plt.figure(k)
        #     a = fig.add_subplot(1, 4, 1)
        #     imgplot = plt.imshow(emphasized[leg_key][k, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 2)
        #     imgplot = plt.imshow(segmented[leg_key][k, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 3)
        #     imgplot = plt.imshow(femur[k, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 4)
        #     imgplot = plt.imshow(segmented[leg_key][k, :, :] - femur[k, :, :], cmap='Greys_r', interpolation="nearest")
        #     if k % 20 == 0:
        #         plt.show()
        # plt.show()

        # import pickle
        # file = open('femur.txt', 'w')
        # pickle.dump(femur, file)
        # file.close()

