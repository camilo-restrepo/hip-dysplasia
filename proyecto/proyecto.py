import dicom
import os
import numpy as np
import pylab
import SimpleITK
import matplotlib.pyplot as plt


PathDicom = "/Volumes/SIN TITULO/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
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

#pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
#pylab.show()


def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)
#    plt.show()


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


reader = SimpleITK.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenamesDICOM)
imgOriginal = reader.Execute()

thresholdFilter = SimpleITK.ThresholdImageFilter()
statsFilter = SimpleITK.StatisticsImageFilter()
otsuFilter = SimpleITK.OtsuMultipleThresholdsImageFilter()
multiplyFilter = SimpleITK.MultiplyImageFilter()

# range(0, imgOriginal.GetDepth()):
ini = 39
end = 42
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

    thresholdFilter.SetLower(stats['mean']+stats['std'])
    thresholdFilter.SetUpper(stats['max'])
    imgFilter = thresholdFilter.Execute(imgMultiply)

    # sitk_show(first)

    fillFilter = SimpleITK.GrayscaleFillholeImageFilter()
    imgFilter = fillFilter.Execute(imgFilter)
    sitk_show(imgFilter)
    #histogram_without_background(imgFilter)

    # addFilter = SimpleITK.AddImageFilter()
    # imgAdd = addFilter.Execute(imgFilter, imgMultiply)

    # sobelFilter = SimpleITK.SobelEdgeDetectionImageFilter()
    # imgFilter = sobelFilter.Execute(imgMultiply)

    # sitk_show(imgFilter)
    # np_show(SimpleITK.GetArrayFromImage(imgFilter))
    #histogram_without_background(imgFilter)

    # imgFilter = addFilter.Execute(imgFilter, imgAdd)

    # statsFilter.Execute(imgFilter)
    # thresholdFilter.SetLower(statsFilter.GetMean()+statsFilter.GetSigma())
    # thresholdFilter.SetUpper(statsFilter.GetMaximum())
    # imgFilter = thresholdFilter.Execute(imgFilter)
    #
    # sitk_show(imgFilter)
    # hist(imgFilter)

plt.show()

# print first.GetPixelIDTypeAsString()
# print imgFilter.GetPixelIDTypeAsString()

    # image = SimpleITK.GetArrayFromImage(imgFilter)
    # u = image[0][0]
    # imgBone = image
    # imgNoBone = image