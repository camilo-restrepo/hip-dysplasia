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

    plt.show()

reader = SimpleITK.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenamesDICOM)
imgOriginal = reader.Execute()

first = imgOriginal[:, :, 40]

minMaxFilter = SimpleITK.MinimumMaximumImageFilter()
minMaxFilter.Execute(first)

first += abs(minMaxFilter.GetMinimum())

imgSmooth = SimpleITK.CurvatureFlow(image1=first,
                                    timeStep=0.125,
                                    numberOfIterations=5)
minMaxFilter.Execute(imgSmooth)

# medianFilter = SimpleITK.MedianImageFilter()
# medianFilter.SetRadius([1, 1, 1])
# imgMedian = medianFilter.Execute(first)

otsuFilter = SimpleITK.OtsuMultipleThresholdsImageFilter()
otsuFilter.SetNumberOfThresholds(2)
imgFilter = otsuFilter.Execute(imgSmooth)

thresholdFilter = SimpleITK.ThresholdImageFilter()
thresholdFilter.SetOutsideValue(0)
thresholdFilter.SetLower(2)
thresholdFilter.SetUpper(2)
imgFilter = thresholdFilter.Execute(imgFilter)
imgFilter /= 2
imgFilter = SimpleITK.Cast(imgFilter, imgSmooth.GetPixelIDValue())

multiplyFilter = SimpleITK.MultiplyImageFilter()
imgFilter = multiplyFilter.Execute(imgFilter, imgSmooth)

# otsuFilter.SetNumberOfThresholds(2)
# imgFilter = otsuFilter.Execute(imgFilter)

sobelFilter = SimpleITK.SobelEdgeDetectionImageFilter()
imgFilter = sobelFilter.Execute(imgFilter)

#sitk_show(imgFilter)

minMaxFilter.Execute(imgFilter)
hist = np.histogram(SimpleITK.GetArrayFromImage(imgFilter), bins=np.arange(minMaxFilter.GetMinimum(),
                                                                           minMaxFilter.GetMaximum()))
hist[0][0] = 0
hist[1][0] = 0
plt.plot(hist[1][:-1], hist[0], lw=2)
plt.show()


# print first.GetPixelIDTypeAsString()
# print imgFilter.GetPixelIDTypeAsString()