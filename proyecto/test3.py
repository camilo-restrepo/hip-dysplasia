import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.filters import sobel


PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
outPath = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/out/"


def np_show(img):
    plt.figure()
    plt.imshow(img, cmap='Greys_r')


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
        elevation_map = sobel(img_original[:, :, i])
        np_show(elevation_map)

plt.show()
