import SimpleITK
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, disk
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import time
import matplotlib.pyplot as plt
import utils
from sklearn import mixture
from multiprocessing import Pool
from multiprocessing import Manager


RIGHT_LEG = 'right_leg'
LEFT_LEG = 'left_leg'
manager = Manager()
windows_cache = manager.dict()
clustering_cache = manager.dict()

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


def remove_noise(sitk_image):
    smooth_filter = SimpleITK.CurvatureFlowImageFilter()
    smooth_filter.SetTimeStep(0.125)
    smooth_filter.SetNumberOfIterations(5)
    img_smooth = smooth_filter.Execute(sitk_image)
    return SimpleITK.GetArrayFromImage(img_smooth)


def load_dicom():
    reader = SimpleITK.ImageSeriesReader()
    filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
    reader.SetFileNames(filenames_dicom)
    img_original = reader.Execute()
    smooth_array = remove_noise(img_original)
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
        # mask[z - 1, x, y],
        # mask[z + 1, x, y]
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


def process_px(px):
    global clustering_cache
    if px not in clustering_cache:
        g = mixture.GMM(n_components=2)
        window = windows_cache[px]
        g.fit(np.reshape(window, (window.size, 1)))

        cntr = g.means_
        v = 0
        if cntr[0, 0] < cntr[1, 0]:
            v = 1

        predict = np.zeros_like(window)
        for z in range(0, window.shape[0]):
            for x in range(0, window.shape[1]):
                for y in range(0, window.shape[2]):
                    predict[z, x, y] = g.predict_proba(window[z, x, y])[0, v]

        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0

        # window_mask = get_mask_window(px)
        # for z in range(0, window.shape[0]):
        #     utils.np_show(predict[z, :, :])
        #     utils.np_show(window_mask[z, :, :])
        #     utils.np_show(window[z, :, :])
        # plt.show()

        # cntr, u, u0, d, jm, h, fpc = cmeans2.cmeans(np.reshape(window, (1, window.size)), 2, 2, 0.001, 1000)
        # v = 0
        # if cntr[0, 0] < cntr[1, 0]:
        #     v = 1
        # result = np.reshape(u[v, :], window.shape)
        # th = threshold_otsu(result)
        # result2 = result > th

        clustering_cache[px] = predict


def compute_boundary():
    mask = legs_bone_masks[leg_key]
    width = mask.shape[1]
    height = mask.shape[2]
    depth = mask.shape[0]
    e_b = set()

    white_pxs = np.argwhere(mask == 1)
    for r in range(0, white_pxs.shape[0]):
        px = white_pxs[r, :]
        z, x, y = px[0], px[1], px[2]
        if 2 < z < depth - 3 and 11 < y < width - 11 and 11 < x < height - 11:
        # if 35 < z < 55 and 11 < y < width - 11 and 11 < x < height - 11:
            if pixel_belongs_to_boundary(x, y, z):
                e_b.add((z, x, y))
                get_window((z, x, y))
                # process_px((z, x, y))
                # break

    return e_b


def replace_volume(point, size=5, depth=2):
    result = legs_bone_masks[leg_key]
    z = point[0]
    x = point[1]
    y = point[2]

    x_ini = x - size
    x_end = x + (size+1)
    y_ini = y - size
    y_end = y + (size+1)
    z_ini = z - depth
    z_end = z + (depth+1)

    # for k in range(0, 5):
    #     utils.np_show(result[z_ini:z_end, x_ini:x_end, y_ini:y_end][k, :, :])
    #     utils.np_show(clustering_cache[point][k, :, :])
    # plt.show()

    result[z_ini:z_end, x_ini:x_end, y_ini:y_end] = clustering_cache[point]


def iterative_adaptative_reclassification():
    boundaries = compute_boundary()
    boundaries_old = np.zeros_like(boundaries)
    bone_mask = legs_bone_masks[leg_key]
    bone_mask_old = bone_mask.copy()
    pool = Pool()
    it = 0
    while not np.all(boundaries == boundaries_old):
        print len(boundaries)
        t0 = time.time()
        pool.map(process_px, boundaries)
        t1 = time.time()
        print '1: ', t1 - t0

        t0 = time.time()
        for px in boundaries:
            replace_volume(px)
        t1 = time.time()
        print '2: ', t1 - t0

        t0 = time.time()
        for z in range(0, bone_mask.shape[0]):
            bone_mask[z, :, :] = closing(bone_mask[z, :, :])
            bone_mask[z, :, :] = ndi.binary_fill_holes(bone_mask[z, :, :])
            bone_mask[z, :, :] = remove_small_objects(bone_mask[z, :, :], 80)

        t1 = time.time()
        print '3: ', t1 - t0

        boundaries_old = boundaries.copy()
        # t0 = time.time()
        # boundaries = compute_boundary()
        # t1 = time.time()
        # print '4: ', t1 - t0
        print len(boundaries), len(boundaries_old), it
        # it += 1

    i = 0
    for k in range(35, 45):
        utils.np_show(bone_mask[k, :, :])
        utils.np_show(bone_mask_old[k, :, :])
        i += 1
        if i == 10:
            plt.show()
            i = 0
    plt.show()


# --------------------------------------------------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------------------------------------------------


img_original_array, img_smooth_array = load_dicom()
legs = get_legs()
emphasized_legs = {}
legs_bone_masks = {}
legs_boundaries = {}

for leg_key in legs.keys():
    if leg_key == RIGHT_LEG:
        emphasized_legs[leg_key] = get_valley_emphasized_image()
        legs_bone_masks[leg_key] = initial_segmentation()
        iterative_adaptative_reclassification()

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

# image = emphasized_legs[leg_key]
# result = np.zeros_like(image)
# threshold = threshold_otsu(image)
# binary_img = image > threshold
# np.multiply(image, binary_img, result)
# result[result == 0] = -1000
#
# # show_img(result)
# pxs = []
# vals = np.argwhere(result != -1000)
# for r in range(0, vals.shape[0]):
#     px = vals[r, :]
#     z, x, y = px[0], px[1], px[2]
#     pxs.append(result[z, x, y])
#
# g = mixture.GMM(n_components=2)
# g.fit(np.reshape(pxs, (len(pxs), 1)))
# print g.means_