import utils
import SimpleITK
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, reconstruction
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border


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
    emphasized_img = image - valley_image
    return emphasized_img


def initial_segmentation(image):
    result = np.zeros_like(image)
    emphasized_img = get_valley_emphasized_image(image)
    threshold = threshold_otsu(emphasized_img)
    binary_img = image > threshold
    np.multiply(image, binary_img, result)
    threshold = threshold_otsu(result)
    binary_img = result > threshold
    for k in range(0, binary_img.shape[0]):
        seed = np.copy(binary_img[k, :, :])
        seed[1:-1, 1:-1] = binary_img[k, :, :].max()
        mask = binary_img[k, :, :]
        binary_img[k, :, :] = reconstruction(seed, mask, method='erosion')
        binary_img[k, :, :] = clear_border(binary_img[k, :, :])
        binary_img[k, :, :] = remove_small_objects(binary_img[k, :, :], 10)

    return binary_img


def pixel_belongs_to_boundary(img_array, x, y, k):
    pixel = img_array[k, x, y]
    if pixel != 0:
        neighbors = [
            img_array[k, x - 1, y - 1],
            img_array[k, x - 1, y],
            img_array[k, x - 1, y + 1],
            img_array[k, x, y - 1],
            img_array[k, x, y + 1],
            img_array[k, x + 1, y - 1],
            img_array[k, x + 1, y],
            img_array[k, x + 1, y + 1],
            img_array[k - 1, x, y],
            img_array[k + 1, x, y]
        ]

        for n in neighbors:
            if n == 0:
                return True
    return False


def compute_boundary(image):
    binary_img = initial_segmentation(image)
    boundaries_array = np.zeros_like(image)
    width = binary_img.shape[1]
    height = binary_img.shape[2]
    depth = binary_img.shape[0]
    e_b = {}

    for k in range(0, depth):
        e_b_list = []
        for index, x in np.ndenumerate(binary_img[k, :, :]):
            if x != 0:
                if 0 < index[0] < width-1 and 0 < index[1] < height-1 and 0 < k < depth-1:
                    if pixel_belongs_to_boundary(binary_img, index[0], index[1], k):
                        boundaries_array[k, index[0], index[1]] = x
                        e_b_list.append(index)
                e_b[k] = e_b_list

    return {'boundaries_array': boundaries_array, 'e_b': e_b}


def iterative_adaptative_reclassification(image):
    result = compute_boundary(image)

    return result['boundaries_array']


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"

img_original = utils.load_dicom(PathDicom)
img_original_array = SimpleITK.GetArrayFromImage(img_original)
img_smooth_array = remove_noise(img_original)

legs = get_legs(img_smooth_array)


ini = 42
end = 49

# ini = 50
# end = img_original.GetDepth()

for leg_key in legs.keys():
    if leg_key == 'right_leg':
        leg = legs[leg_key]
        # result = get_valley_emphasized_image(leg)
        # v = result['v']
        # e = result['e']
        # seg = segmentation(leg)
        seg = iterative_adaptative_reclassification(leg)
        i = 0
        for z in range(0, img_original.GetDepth()):
            if ini <= z <= end:
                utils.np_show(leg[z, :, :])
                utils.np_show(seg[z, :, :])
                i += 1
                if i == 10:
                    plt.show()
                    i = 0


plt.show()
