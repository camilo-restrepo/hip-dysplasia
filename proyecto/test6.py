import SimpleITK
import matplotlib.pyplot as plt
import utils
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, convex_hull_object
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


def get_sides_center_coordinates(sitk_image):
    img_smooth = remove_noise(sitk_image)
    bone = np.zeros_like(img_smooth)
    bone[img_smooth < 200] = 0
    bone[img_smooth > 200] = 1
    bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
    label_image = label(bone)

    centroids = []
    for region in regionprops(label_image):
        centroids.append(region.centroid)
    return centroids


def get_bone_mask(image_array):
    body_threshold = threshold_otsu(image_array)
    body = image_array > body_threshold
    bone = np.multiply(body, image_array)

    # bone_threshold = threshold_otsu(body)
    # bone = body > bone_threshold

    # bone = np.copy(bone)
    # for k in range(0, bone.shape[0]):
    #     b = bone[k, :, :]
    #     b = ndi.binary_fill_holes(b.astype(bool))
    #     bone[k, :, :] = remove_small_objects(b.astype(bool), 200)

    #
    # label_image = label(bone)
    # mid = label_image[:, 256]
    # labels = set(mid[mid != 0])
    #
    # for i in range(0, label_image.shape[0]):
    #     for j in range(0, label_image.shape[1]):
    #         if label_image[i, j] in labels:
    #             bone[i, j] = 0
    #             label_image[i, j] = 0

    return bone


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"


reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()
img_original_array = SimpleITK.GetArrayFromImage(img_original)

remove_small_objects_size = 80
# obtener izq y der separados y mas pequenos
centroids = get_sides_center_coordinates(img_original[:, :, 0])

row1 = centroids[0][0].astype(int)
row2 = centroids[1][0].astype(int)
row_ini = 0
if row1 - 128 > 0:
    row_ini = row1 - 128
row_end = row_ini + 256
right_leg = img_original_array[:, row_ini:row_end, 0:256]

row_ini = 0
if row1 - 128 > 0:
    row_ini = row2 - 128
row_end = row_ini + 256
left_leg = img_original_array[:, row_ini:row_end, 256:]

# ESTO TIENE QUE SER ASI PARA PRUEBAS SOLO DER: legs = [right_leg, left_leg]
legs = [right_leg]

ini = 42
end = 49
for leg in legs:
    leg_smooth_array = remove_noise(SimpleITK.GetImageFromArray(leg))
    mask = get_bone_mask(leg_smooth_array)

    for z in range(0, img_original.GetDepth()):
        if ini <= z <= end:
            # r = np.multiply(mask[z, :, :], leg_smooth_array[z, :, :])
            r = mask[z, :, :]
            r[r < 0] = 0
            c = centroids[0]
            init = np.array([c[0], c[1]]).T
            utils.np_show(r)
            snake = active_contour(r, init, alpha=0.015, beta=10, gamma=0.001)
            utils.np_show(snake)


plt.show()




















