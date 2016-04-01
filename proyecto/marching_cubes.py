import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

img_smooth = smooth_filter.Execute(img_original)
img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)
verts, faces = measure.marching_cubes(img_smooth_array, 300)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection(verts[faces])
ax.add_collection3d(mesh)

# ini = 37
# end = 37
# for i in range(0, img_original.GetDepth()):
#     if ini <= i <= end:
#

plt.show()
