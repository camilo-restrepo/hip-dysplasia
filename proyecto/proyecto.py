import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage import morphology

# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# outPath = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/out/"

PathDicom = "D:\imagenes\ALMANZA_RUIZ_JUAN_CARLOS\TAC_DE_PELVIS - 84441\_Bone_30_2"
outPath = "D:\imagenes\out"

# lstFilesDCM = []
# for dirName, subdirList, fileList in os.walk(PathDicom):
#     for filename in fileList:
#         if ".dcm" in filename.lower():  # check whether the file's DICOM
#             lstFilesDCM.append(os.path.join(dirName, filename))
#
# RefDs = dicom.read_file(lstFilesDCM[0])
#
# print RefDs.RescaleSlope
# print RefDs.RescaleIntercept

# ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
# ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

# print RefDs
# print ConstPixelDims
# print ConstPixelSpacing

# x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
# y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
# z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
#
# ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
#
# for filenameDCM in lstFilesDCM:
#     ds = dicom.read_file(filenameDCM)
#     ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

# pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
# pylab.show()


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
threshold_filter = SimpleITK.ThresholdImageFilter()
threshold_filter.SetOutsideValue(0)
otsu_filter = SimpleITK.OtsuMultipleThresholdsImageFilter()
multiply_filter = SimpleITK.MultiplyImageFilter()
binary_threshold_filter = SimpleITK.BinaryThresholdImageFilter()
binary_threshold_filter.SetInsideValue(1)
binary_threshold_filter.SetOutsideValue(0)
morphological_filter = SimpleITK.GrayscaleMorphologicalClosingImageFilter()
morphological_filter.SetKernelRadius([1, 1])
grayscale_fill_filter = SimpleITK.GrayscaleFillholeImageFilter()
binary_fill_filter = SimpleITK.BinaryFillholeImageFilter()
medianFilter = SimpleITK.MedianImageFilter()
medianFilter.SetRadius([2, 2, 2])
# DISTRIBUCION DEL RUIDO OBTENIDA CON IMAGEJ: Mean: -1021.905 Std: 44.194


def initial_binary_threshold(image):
    # Segmentacion del cuerpo
    img_smooth = smooth_filter.Execute(image)
    otsu_filter.SetNumberOfThresholds(2)
    img_filter = otsu_filter.Execute(img_smooth)
    threshold_filter.SetLower(2)
    threshold_filter.SetUpper(2)
    img_filter = threshold_filter.Execute(img_filter)
    img_filter /= 2
    img_filter = SimpleITK.Cast(img_filter, img_smooth.GetPixelIDValue())
    img_multiply = multiply_filter.Execute(img_filter, img_smooth)

    # Segmentacion de los huesos
    otsu_filter.SetNumberOfThresholds(1)
    img_filter = otsu_filter.Execute(img_multiply)

    # stats = get_stats_without_background(img_multiply)
    # threshold_filter.SetLower(stats['mean'] + stats['std'])
    # threshold_filter.SetUpper(float(stats['max']))
    # img_filter = threshold_filter.Execute(img_multiply)
    # stats = get_stats_without_background(img_filter)
    # binary_threshold_filter.SetLowerThreshold(stats['min'])
    # binary_threshold_filter.SetUpperThreshold(stats['max'])
    # img_filter = binary_threshold_filter.Execute(img_filter)

    # Rellenar huecos
    img_filter = morphological_filter.Execute(img_filter)
    img_filter = grayscale_fill_filter.Execute(img_filter)
    img_filter = SimpleITK.Cast(img_filter, image.GetPixelIDValue())
    return img_filter


def get_segmented_image(original):
    mask = initial_binary_threshold(original)
    # mask = SimpleITK.Cast(mask, original.GetPixelIDValue())
    mask.CopyInformation(original)
    m = multiply_filter.Execute(original, mask)
    threshold_filter.SetOutsideValue(0)
    stats = get_stats(m)
    threshold_filter.SetLower(0)
    threshold_filter.SetUpper(float(stats['max']))
    m = threshold_filter.Execute(m)
    return m

thresholded_ct_scan_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))

for i in range(0, img_original.GetDepth()):
    thresholded_ct_scan_array[:, :, i] = SimpleITK.GetArrayFromImage(initial_binary_threshold(img_original[:, :, i]))

boundaries = compute_boundary(thresholded_ct_scan_array)

ini = 37
end = 37
for i in range(0, img_original.GetDepth()):
    if ini <= i <= end:
        # temp = initial_binary_threshold(img_original[:, :, i])
        temp = SimpleITK.GetImageFromArray(thresholded_ct_scan_array[:, :, i])
        sitk_show(temp)

        rem = morphology.remove_small_objects(thresholded_ct_scan_array[:, :, i], 5)
        temp = SimpleITK.GetImageFromArray(rem)
        sitk_show(temp)
        sitk_show(img_original[:, :, i])
        np_show(boundaries['boundaries_array'][:, :, i])
        # sitk_show(SimpleITK.Tile(temp, img_original[:, :, i], (2, 1, 0)))

plt.show()

    # ------------------- Adaptative Thresholding Segmentation --------------------
    # previous_error = 0
    # while True:
    #     boundary = compute_boundary(imgFilter)
    #     error = recalculate_segmentation(imgFilter, boundary['e_b'])
    #     imagen = SimpleITK.GetArrayFromImage(imgFilter)
    #     for e in error:
    #         imagen[e[0]][e[1]] = 0
    #     imgFilter = SimpleITK.GetImageFromArray(imagen)
    #     if len(error) == 0 or len(error) == previous_error:
    #         break
    #     previous_error = len(error)
    # sitk_show(imgFilter)
    # -----------------------------------------------------------------------------
    #

# print first.GetPixelIDTypeAsString()
# print imgFilter.GetPixelIDTypeAsString()

# Scikit Image
# for i in range(0, imgOriginal.GetDepth()):
#     tifffile.imsave(outPath+'test-'+'{:03d}'.format(i)+'.tif', boundaries['boundaries_array'][:, :, i])
