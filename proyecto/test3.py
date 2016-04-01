import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import morphology
from skimage.segmentation import random_walker
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed


# PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"


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
end = 37
for i in range(0, img_original.GetDepth()):
    if ini <= i <= end:
        img_smooth = smooth_filter.Execute(img_original[:, :, i])
        img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)
        distance = ndi.distance_transform_edt(img_smooth_array)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                    labels=img_smooth_array)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=img_smooth_array)

        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax0, ax1, ax2 = axes

        ax0.imshow(img_smooth_array, cmap=plt.cm.gray, interpolation='nearest')
        ax0.set_title('Overlapping objects')
        ax1.imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
        ax1.set_title('Distances')
        ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
        ax2.set_title('Separated objects')

        for ax in axes:
            ax.axis('off')

        fig.tight_layout()




        #
        # elevation_map = sobel(img_smooth_array)
        #
        # markers = np.zeros_like(img_smooth_array)
        # markers[img_smooth_array < 100] = 1
        # markers[img_smooth_array > 300] = 2
        # segmentation = morphology.watershed(elevation_map, markers)
        # segmentation = ndi.binary_fill_holes(segmentation - 1)
        # np_show(segmentation)
        # labels = random_walker(img_smooth_array, markers, beta=10, mode='bf')
        # np_show(labels)

plt.show()
