import SimpleITK
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, disk, opening
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import time
import matplotlib.pyplot as plt
import utils
from sklearn import mixture
from multiprocessing import Pool, Manager


RIGHT_LEG = 'right_leg'
LEFT_LEG = 'left_leg'
windows_cache = Manager().dict()
prediction_cache = dict()

PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"


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


# --------------------------------------------------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------------------------------------------------


def remove_noise_curvature_flow(sitk_image):
    smooth_filter = SimpleITK.CurvatureFlowImageFilter()
    smooth_filter.SetTimeStep(0.125)
    smooth_filter.SetNumberOfIterations(5)
    img_smooth = smooth_filter.Execute(sitk_image)
    return SimpleITK.GetArrayFromImage(img_smooth)


def remove_noise_anisotropic(sitk_image):
    # TODO
    smooth_filter = SimpleITK.GradientAnisotropicDiffusionImageFilter()
    smooth_filter.SetTimeStep(0.125)
    smooth_filter.SetNumberOfIterations(5)
    img_smooth = smooth_filter.Execute(sitk_image)
    return SimpleITK.GetArrayFromImage(img_smooth)


def load_dicom():
    reader = SimpleITK.ImageSeriesReader()
    filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
    reader.SetFileNames(filenames_dicom)
    img_original = reader.Execute()
    smooth_array = remove_noise_curvature_flow(img_original)
    original_array = SimpleITK.GetArrayFromImage(img_original)
    return original_array, smooth_array


# --------------------------------------------------------------------------------------------------------------------
# SEPARATE LEGS
# --------------------------------------------------------------------------------------------------------------------


def get_sides_center_coordinates(remove_small_objects_size=80):
    bone = np.zeros_like(img_smooth_array[0, :, :])
    bone[img_smooth_array[0, :, :] < 200] = 0
    bone[img_smooth_array[0, :, :] > 200] = 1

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
    right_leg = img_smooth_array[:, rows['row_ini']:rows['row_end'], 0:256]
    # right_leg = img_original_array[:, rows['row_ini']:rows['row_end'], 0:256]

    row2 = centroids[1][0].astype(int)
    rows = get_rows(row2)
    left_leg = img_smooth_array[:, rows['row_ini']:rows['row_end'], 256:]
    # left_leg = img_original_array[:, rows['row_ini']:rows['row_end'], 256:]

    return {RIGHT_LEG: right_leg, LEFT_LEG: left_leg}


# --------------------------------------------------------------------------------------------------------------------
# VALLEY COMPUTATION
# --------------------------------------------------------------------------------------------------------------------


def get_valley_image():
    image = legs[leg_key]
    selem = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
    valley_img = closing(image, selem)
    valley_img = valley_img - image
    return valley_img


def get_valley_emphasized_image():
    image = legs[leg_key]
    valley_image = get_valley_image()
    valley_image = image - valley_image
    return valley_image


# --------------------------------------------------------------------------------------------------------------------
# INITIAL SEGMENTATION
# --------------------------------------------------------------------------------------------------------------------


def initial_segmentation():
    image = emphasized_legs[leg_key]
    result = np.zeros_like(image)
    threshold = threshold_otsu(image)
    binary_img = image > threshold
    np.multiply(image, binary_img, result)

    for z in range(0, image.shape[0]):
        threshold = threshold_otsu(result[z, :, :])
        binary_img[z, :, :] = result[z, :, :] > threshold
        binary_img[z, 240:, :] = 0
        img = binary_img[z, :, :].astype(int)
        img = closing(img, disk(2))
        img = ndi.binary_fill_holes(img)
        binary_img[z, :, :] = remove_small_objects(img, 80)
    return binary_img


# --------------------------------------------------------------------------------------------------------------------
# ITERATIVE ADAPTATIVE RECLASSIFICATION
# --------------------------------------------------------------------------------------------------------------------


def pixel_belongs_to_boundary(x, y, z):
    mask = legs_bone_masks[leg_key]
    neighbors = np.array([
        # mask[z, x - 1, y - 1],
        mask[z, x - 1, y],
        # mask[z, x - 1, y + 1],
        mask[z, x, y - 1],
        mask[z, x, y + 1],
        # mask[z, x + 1, y - 1],
        mask[z, x + 1, y],
        # mask[z, x + 1, y + 1],
        mask[z - 1, x, y],
        mask[z + 1, x, y]
    ])
    return np.any(neighbors == 0)


def has_black_neighbors(z, x, y):
    mask = legs_bone_masks[leg_key]
    neighbors = np.array([
        mask[z, x - 1, y - 1],
        mask[z, x - 1, y],
        mask[z, x - 1, y + 1],
        mask[z, x, y - 1],
        mask[z, x, y + 1],
        mask[z, x + 1, y - 1],
        mask[z, x + 1, y],
        mask[z, x + 1, y + 1]
    ])
    return np.any(neighbors == 0)


def get_window(point, size=5, depth=2):
    global windows_cache
    z = point[0]
    x = point[1]
    y = point[2]
    if (z, x, y) not in windows_cache:
        image = emphasized_legs[leg_key]
        x_ini = x - size
        x_end = x + (size+1)
        y_ini = y - size
        y_end = y + (size+1)
        z_ini = z - depth
        z_end = z + (depth+1)
        window = image[z_ini:z_end, x_ini:x_end, y_ini:y_end]
        windows_cache[(z, x, y)] = window


def get_mask_window(point, size=5, depth=2):
    image = legs_bone_masks[leg_key]
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


def compute_px_model(px):
    process = px[3]
    px = (px[0], px[1], px[2])
    if not process:
        g = mixture.GMM(n_components=2)
        window = windows_cache[px]

        # diamond = []
        # diamond.extend(np.reshape(window[2, :, :], (-1,)))
        # diamond.append(window[0, 5, 5])
        # diamond.append(window[4, 5, 5])
        # diamond.extend(np.reshape(window[1, 3:8, 3:8], (-1,)))
        # diamond.extend(np.reshape(window[3, 3:8, 3:8], (-1,)))
        # g.fit(np.reshape(diamond, (len(diamond), 1)))
        g.fit(np.reshape(window, (window.size, 1)))

        cntr = g.means_
        v = 0
        if cntr[0, 0] < cntr[1, 0]:
            v = 1

        predictions = g.predict_proba(np.reshape(window, (window.size, 1)))
        predictions = np.reshape(predictions[:, v], window.shape)

        predictions = predictions[2, :, :]
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0
        return (px, predictions)
    return (px, None)


def compute_boundary():
    mask = legs_bone_masks[leg_key]
    width = mask.shape[1]
    height = mask.shape[2]
    depth = mask.shape[0]
    e_b = set()
    e_b2 = set()

    white_pxs = np.argwhere(mask == 1)
    for r in range(0, white_pxs.shape[0]):
        px = white_pxs[r, :]
        z, x, y = px[0], px[1], px[2]
        # if 2 < z < depth - 3 and 11 < y < width - 11 and 11 < x < height - 11:
        if 35 < z < 55 and 11 < y < width - 11 and 11 < x < height - 11:
            # if not has_black_neighbors(z, x, y):
            if pixel_belongs_to_boundary(x, y, z):
                e_b.add((z, x, y))
                e_b2.add((z, x, y, (z, x, y) in prediction_cache))
                get_window((z, x, y))

    return e_b, e_b2


def replace_volume(point, size=5, depth=2):
    global legs_bone_masks
    z = point[0]
    x = point[1]
    y = point[2]

    x_ini = x - size
    x_end = x + (size+1)
    y_ini = y - size
    y_end = y + (size+1)
    z_ini = z - depth
    z_end = z + (depth+1)

    # result[z_ini:z_end, x_ini:x_end, y_ini:y_end] = prediction_cache[point]
    legs_bone_masks[leg_key][z, x_ini:x_end, y_ini:y_end] = prediction_cache[point]


def iterative_adaptative_reclassification():
    global prediction_cache
    global legs_bone_masks
    boundaries, boundaries2 = compute_boundary()
    boundaries_old = np.zeros_like(boundaries)
    bone_mask_old = legs_bone_masks[leg_key].copy()
    pool = Pool()
    it = 0
    selem = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
    while not np.all(boundaries == boundaries_old):
        print len(boundaries)
        t0 = time.time()
        predictions = pool.map(compute_px_model, boundaries2)
        for prediction in predictions:
            if prediction[0] not in prediction_cache:
                prediction_cache[prediction[0]] = prediction[1]
        t1 = time.time()
        print '1: ', t1 - t0

        last = legs_bone_masks[leg_key].copy()

        t0 = time.time()
        for px in boundaries:
            replace_volume(px)
        t1 = time.time()
        print '2: ', t1 - t0

        t0 = time.time()
        # legs_bone_masks[leg_key] = closing(legs_bone_masks[leg_key], selem=selem)
        # legs_bone_masks[leg_key] = opening(legs_bone_masks[leg_key], selem=selem)

        for z in range(0, legs_bone_masks[leg_key].shape[0]):
            legs_bone_masks[leg_key][z, :, :] = closing(legs_bone_masks[leg_key][z, :, :])
            legs_bone_masks[leg_key][z, :, :] = ndi.binary_fill_holes(legs_bone_masks[leg_key][z, :, :])
            legs_bone_masks[leg_key][z, :, :] = remove_small_objects(legs_bone_masks[leg_key][z, :, :], 80)

        t1 = time.time()
        print '3: ', t1 - t0

        boundaries_old = boundaries.copy()
        t0 = time.time()
        boundaries, boundaries2 = compute_boundary()
        t1 = time.time()
        print '4: ', t1 - t0

        it += 1
        print len(boundaries), len(boundaries_old), it

        if it % 2 == 0:
            diff = legs_bone_masks[leg_key] - last
            for k in range(35, 45):
                fig = plt.figure(k)
                a = fig.add_subplot(1, 4, 1)
                imgplot = plt.imshow(legs_bone_masks[leg_key][k, :, :], cmap='Greys_r', interpolation="nearest")
                a = fig.add_subplot(1, 4, 2)
                imgplot = plt.imshow(last[k, :, :], cmap='Greys_r', interpolation="nearest")
                a = fig.add_subplot(1, 4, 3)
                imgplot = plt.imshow(diff[k, :, :], cmap='Greys_r', interpolation="nearest")
                a = fig.add_subplot(1, 4, 4)
                imgplot = plt.imshow(bone_mask_old[k, :, :], cmap='Greys_r', interpolation="nearest")
            plt.show()


# --------------------------------------------------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------------------------------------------------


img_original_array, img_smooth_array = load_dicom()
legs = get_legs()
emphasized_legs = {}
legs_bone_masks = {}
legs_boundaries = {}
valleys = {}


for leg_key in legs.keys():
    if leg_key == RIGHT_LEG:
        emphasized_legs[leg_key] = get_valley_emphasized_image()
        # legs_bone_masks[leg_key] = initial_segmentation()
        valleys[leg_key] = get_valley_image()

        for k in range(35, 45):
            fig = plt.figure(k)
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(emphasized_legs[leg_key][k, :, :], cmap='Greys_r', interpolation="nearest")
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(valleys[leg_key][k, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 3)
        #     imgplot = plt.imshow(emphasized_legs[leg_key][k, :, :], cmap='Greys_r', interpolation="nearest")
        #     a = fig.add_subplot(1, 4, 4)
        #     # imgplot = plt.imshow(v_bin[k, :, :], cmap='Greys_r', interpolation="nearest")
        plt.show()

        # iterative_adaptative_reclassification()

        # i = 0
        # for k in range(35, 55):
        #     utils.np_show(emphasized_legs[leg_key][k, :, :])
        #     utils.np_show(legs_bone_masks[leg_key][k, :, :])
        #     i += 1
        #     if i == 10:
        #         plt.show()
        #         i = 0
        # plt.show()
        # ta = time.time()
        # b = compute_boundary()
        # print len(b)
        # tb = time.time()
        # print tb - ta
        # print len(b)
        # test = np.zeros_like(emphasized_legs[leg_key])
        # for v in b:
        #     test[v[0], v[1], v[2]] = 1
        # show_img(test)
