# import dicom
# import os
# import pylab
import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"


# lstFilesDCM = []
# for dirName, subdirList, fileList in os.walk(PathDicom):
#     for filename in fileList:
#         if ".dcm" in filename.lower():  # check whether the file's DICOM
#             lstFilesDCM.append(os.path.join(dirName, filename))

# print lstFilesDCM

# RefDs = dicom.read_file(lstFilesDCM[0])
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


def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)


# plt.show()


def np_show(img):
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


def pixel_belongs_to_boundary(imgArray, x, y, z):
    pixel = imgArray[x][y]
    neighbors = [
        imgArray[x - 1][y - 1],
        imgArray[x - 1][y],
        imgArray[x - 1][y + 1],
        imgArray[x][y - 1],
        imgArray[x][y + 1],
        imgArray[x + 1][y - 1],
        imgArray[x + 1][y],
        imgArray[x + 1][y + 1]
    ]

    for n in neighbors:
        if pixel == 0 and n != 0:
            return True
        elif pixel != 0 and n == 0:
            return True
        return False

    return False


def compute_boundary(img):
    imgArray = SimpleITK.GetArrayFromImage(img)
    e_b = np.zeros_like(imgArray)
    width = img.GetWidth()
    height = img.GetHeight()
    e_b_list = []
    for index, x in np.ndenumerate(imgArray):
        if 0 < index[0] < width-1 and 0 < index[1] < height-1:
            if pixel_belongs_to_boundary(imgArray, index[0], index[1], 0):
                e_b[index[0]][index[1]] = x
                e_b_list.append(index)
    return {'img': e_b, 'e_b': e_b_list}


def recalculate_segmentation(img, e_b):
    width = img.GetWidth()
    height = img.GetHeight()
    original = SimpleITK.GetArrayFromImage(img)
    segmentation_error = []
    clf = GaussianNB()

    for px in e_b:
        x = px[0]
        y = px[1]
        if 0 < x < width-1 and 0 < y < height-1:
            y_real = []
            neighbors = [
                original[x - 1][y - 1],
                original[x - 1][y],
                original[x - 1][y + 1],
                original[x][y - 1],
                original[x][y + 1],
                original[x + 1][y - 1],
                original[x + 1][y],
                original[x + 1][y + 1]
            ]
            for n in neighbors:
                if n != 0:
                    y_real.append(1)
                else:
                    y_real.append(0)

            neighbors = np.reshape(neighbors, (len(neighbors), 1))

            clf.fit(neighbors, y_real)
            y_predict = clf.predict(original[x][y])[0]

            # (y_predict == 0 and original[x][y] != 0) or (y_predict == 1 and original[x][y] == 0)
            if y_predict == 0 and original[x][y] != 0:
                segmentation_error.append(px)
    return segmentation_error


reader = SimpleITK.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenamesDICOM)
imgOriginal = reader.Execute()

thresholdFilter = SimpleITK.ThresholdImageFilter()
statsFilter = SimpleITK.StatisticsImageFilter()
otsuFilter = SimpleITK.OtsuMultipleThresholdsImageFilter()
multiplyFilter = SimpleITK.MultiplyImageFilter()

# range(0, imgOriginal.GetDepth()):
ini = 87
end = 90
for i in range(ini, end):
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetLower(2)
    thresholdFilter.SetUpper(2)
    first = imgOriginal[:, :, i]

    statsFilter.Execute(first)
    first += abs(statsFilter.GetMinimum())
    imgSmooth = SimpleITK.CurvatureFlow(image1=first, timeStep=0.125, numberOfIterations=5)

    # medianFilter = SimpleITK.MedianImageFilter()
    # medianFilter.SetRadius([1, 1, 1])
    # imgMedian = medianFilter.Execute(first)

    otsuFilter.SetNumberOfThresholds(2)
    imgFilter = otsuFilter.Execute(imgSmooth)
    imgFilter = thresholdFilter.Execute(imgFilter)
    imgFilter /= 2
    imgFilter = SimpleITK.Cast(imgFilter, imgSmooth.GetPixelIDValue())
    imgMultiply = multiplyFilter.Execute(imgFilter, imgSmooth)

    stats = get_stats_without_background(imgMultiply)

    thresholdFilter.SetLower(stats['mean'] + stats['std'])
    thresholdFilter.SetUpper(stats['max'])
    imgFilter = thresholdFilter.Execute(imgMultiply)

    # sitk_show(first)

    fillFilter = SimpleITK.GrayscaleFillholeImageFilter()
    imgFilter = fillFilter.Execute(imgFilter)
    sitk_show(imgFilter)

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
    # histogram_without_background(imgFilter)


plt.show()

# print first.GetPixelIDTypeAsString()
# print imgFilter.GetPixelIDTypeAsString()

# image = SimpleITK.GetArrayFromImage(imgFilter)
# u = image[0][0]
# imgBone = image
# imgNoBone = image
