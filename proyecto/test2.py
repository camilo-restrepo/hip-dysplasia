import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, reconstruction, remove_small_objects

PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
outPath = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/out/"

# PathDicom = "D:\imagenes\ALMANZA_RUIZ_JUAN_CARLOS\TAC_DE_PELVIS - 84441\_Bone_30_2"
# outPath = "D:\imagenes\out"


def sitk_show(img):
    img_array = SimpleITK.GetArrayFromImage(img)
    plt.figure()
    plt.imshow(img_array, cmap='Greys_r')


def np_show(img):
    plt.figure()
    plt.imshow(img, cmap='Greys_r')


def histogram_without_background(img):
    hist_filter = SimpleITK.MinimumMaximumImageFilter()
    hist_filter.Execute(img)
    histogram = np.histogram(SimpleITK.GetArrayFromImage(img), bins=np.arange(hist_filter.GetMinimum(),
                                                                              hist_filter.GetMaximum()))
    plt.figure(2)
    histogram[0][0] = 0
    histogram[1][0] = 0
    plt.plot(histogram[1][:-1], histogram[0], lw=1)


def get_stats_without_background(img):
    img_array = SimpleITK.GetArrayFromImage(img)
    img_array = img_array[img_array > 0]
    return {"mean": np.mean(img_array), "std": np.std(img_array), "max": np.max(img_array), "min": np.min(img_array)}


def get_stats(img):
    img_array = SimpleITK.GetArrayFromImage(img)
    return {"mean": np.mean(img_array), "std": np.std(img_array), "max": np.max(img_array), "min": np.min(img_array)}


def pixel_belongs_to_boundary(img_array, x, y, z):
    pixel = img_array[x, y, z]
    if pixel != 0:
        neighbors = [
            img_array[x - 1][y - 1][z],
            img_array[x - 1][y][z],
            img_array[x - 1][y + 1][z],
            img_array[x][y - 1][z],
            img_array[x][y + 1][z],
            img_array[x + 1][y - 1][z],
            img_array[x + 1][y][z],
            img_array[x + 1][y + 1][z],
            img_array[x][y][z-1],
            img_array[x][y][z+1],
        ]

        for n in neighbors:
            if n == 0:
                return True
    return False


def compute_boundary(ct_array):
    width = ct_array.shape[0]
    height = ct_array.shape[1]
    depth = ct_array.shape[2]
    boundaries_array = np.zeros((width, height, depth))
    e_b = {}

    for z in range(0, depth):
        e_b_list = []
        for index, x in np.ndenumerate(ct_array[:, :, z]):
            if x != 0:
                if 0 < index[0] < width-1 and 0 < index[1] < height-1 and 0 < z < depth-1:
                    if pixel_belongs_to_boundary(ct_array, index[0], index[1], z):
                        boundaries_array[index[0], index[1], z] = x
                        e_b_list.append(index)
                e_b[z] = e_b_list

    return {'boundaries_array': boundaries_array, 'e_b': e_b}


# def recalculate_segmentation(ct_array, e_b):
#     width = ct_array.shape[0]
#     height = ct_array.shape[1]
#     depth = ct_array.shape[2]
#
#     segmentation_error = []
#     clf = GaussianNB()
#
#     for px in e_b:
#         x = px[0]
#         y = px[1]
#         if 0 < x < width-1 and 0 < y < height-1:
#             y_real = []
#             neighbors = [
#                 original[x - 1][y - 1],
#                 original[x - 1][y],
#                 original[x - 1][y + 1],
#                 original[x][y - 1],
#                 original[x][y + 1],
#                 original[x + 1][y - 1],
#                 original[x + 1][y],
#                 original[x + 1][y + 1]
#             ]
#             for n in neighbors:
#                 if n != 0:
#                     y_real.append(1)
#                 else:
#                     y_real.append(0)
#
#             neighbors = np.reshape(neighbors, (len(neighbors), 1))
#
#             clf.fit(neighbors, y_real)
#             y_predict = clf.predict(original[x][y])[0]
#
#             # (y_predict == 0 and original[x][y] != 0) or (y_predict == 1 and original[x][y] == 0)
#             if y_predict == 0 and original[x][y] != 0:
#                 segmentation_error.append(px)
#     return segmentation_error


reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()

smooth_filter = SimpleITK.CurvatureFlowImageFilter()
smooth_filter.SetTimeStep(0.125)
smooth_filter.SetNumberOfIterations(5)

closing_radius = disk(1)
remove_small_objects_size = 20


def get_bone_mask(image):
    # Segmentacion del cuerpo
    threshold = threshold_otsu(image)
    binary = image > threshold
    binary = np.multiply(binary, image)

    # Segmentacion de los huesos
    threshold = threshold_otsu(binary)
    binary = binary > threshold

    # Rellenar huecos
    binary = closing(binary, closing_radius)
    binary = remove_small_objects(binary, remove_small_objects_size)

    # Llenar huecos
    seed = np.copy(binary)
    seed[1:-1, 1:-1] = binary.max()
    mask = binary
    filled = reconstruction(seed, mask, method='erosion')
    return filled


def get_segmented_image(image):
    img_smooth = smooth_filter.Execute(image)
    img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)
    img_array = SimpleITK.GetArrayFromImage(image)
    mask = get_bone_mask(img_smooth_array)
    img_array = np.multiply(img_array, mask)
    img_array[img_array < 0] = 0
    return img_array

# thresholded_ct_scan_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))
#
#
# for i in range(0, img_original.GetDepth()):
#     thresholded_ct_scan_array[:, :, i] = SimpleITK.GetArrayFromImage(get_threshold_mask(img_original[:, :, i]))
#     tmp = morphology.remove_small_objects(thresholded_ct_scan_array[:, :, i].astype(bool), remove_small_objects_size)
#     thresholded_ct_scan_array[:, :, i] = tmp.astype(int)
#
# boundaries = compute_boundary(thresholded_ct_scan_array)

ini = 37
end = 42
for i in range(0, img_original.GetDepth()):
    if ini <= i <= end:
        thresholded_img = get_segmented_image(img_original[:, :, i])
        sitk_show(img_original[:, :, i])
        np_show(thresholded_img)

plt.show()


# Scikit Image
# for i in range(0, imgOriginal.GetDepth()):
#     tifffile.imsave(outPath+'test-'+'{:03d}'.format(i)+'.tif', boundaries['boundaries_array'][:, :, i])


# DISTRIBUCION DEL RUIDO OBTENIDA CON IMAGEJ: Mean: -1021.905 Std: 44.194