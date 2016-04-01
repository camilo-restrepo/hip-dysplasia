import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import morphology
from skimage.segmentation import random_walker
from scipy import ndimage as ndi


PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"


def np_show(img):
    plt.figure()
    plt.imshow(img, cmap='Greys_r')


def sitk_show(img):
    img_array = SimpleITK.GetArrayFromImage(img)
    plt.figure()
    plt.imshow(img_array, cmap='Greys_r')

reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()

smooth_filter = SimpleITK.CurvatureFlowImageFilter()
smooth_filter.SetTimeStep(0.125)
smooth_filter.SetNumberOfIterations(5)

ini = 37
end = 42
for i in range(0, img_original.GetDepth()):
    if ini <= i <= end:
        img_smooth = smooth_filter.Execute(img_original[:, :, i])
        img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)
        elevation_map = sobel(img_smooth_array)

        markers = np.zeros_like(img_smooth_array)
        markers[img_smooth_array < 100] = 1
        markers[img_smooth_array > 300] = 2
        segmentation = morphology.watershed(elevation_map, markers)
        segmentation = ndi.binary_fill_holes(segmentation - 1)
        np_show(segmentation)
        labels = random_walker(img_smooth_array, markers, beta=10, mode='bf')
        np_show(labels)

plt.show()
