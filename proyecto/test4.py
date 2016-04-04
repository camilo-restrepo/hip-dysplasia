import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, reconstruction, remove_small_objects
import utils


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# outPath = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/out/"

# PathDicom = "D:\imagenes\ALMANZA_RUIZ_JUAN_CARLOS\TAC_DE_PELVIS - 84441\_Bone_30_2"
# outPath = "D:\imagenes\out"

reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()

smooth_filter = SimpleITK.CurvatureFlowImageFilter()
smooth_filter.SetTimeStep(0.125)
smooth_filter.SetNumberOfIterations(5)

closing_radius = disk(1)
remove_small_objects_size = 80


def get_bone_mask(image):
    # Remove noise
    img_smooth = smooth_filter.Execute(image)
    img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)

    bone = np.zeros_like(img_smooth_array)
    bone[img_smooth_array < 150] = 0
    bone[img_smooth_array > 150] = 1
    bone = closing(bone, closing_radius)
    bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
    seed = np.copy(bone)
    seed[1:-1, 1:-1] = bone.max()
    mask = bone
    bone = reconstruction(seed, mask, method='erosion')

    return bone


def get_segmented_image(image):
    img_array = SimpleITK.GetArrayFromImage(image)
    mask = get_bone_mask(image)
    img_array = np.multiply(img_array, mask)
    img_array[img_array < 0] = 0
    return img_array


# mask_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))
# thresholded_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))

# for i in range(0, img_original.GetDepth()):
    # mask_array[:, :, i] = get_bone_mask(img_original[:, :, i])
    # utils.np_show(mask_array[:, :, i])
    # thresholded_array[:, :, i] = get_segmented_image(img_original[:, :, i])
    # utils.np_show(thresholded_array[:, :, i])


ini = 37
end = 42
for i in range(0, img_original.GetDepth()):
    if ini <= i <= end:
        r = get_segmented_image(img_original[:, :, i])
        utils.np_show(r)
        # utils.show_hist(r)

plt.show()

# Scikit Image
# for i in range(0, imgOriginal.GetDepth()):
#     tifffile.imsave(outPath+'test-'+'{:03d}'.format(i)+'.tif', boundaries['boundaries_array'][:, :, i])


# DISTRIBUCION DEL RUIDO OBTENIDA CON IMAGEJ: Mean: -1021.905 Std: 44.194
