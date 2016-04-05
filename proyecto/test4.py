import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, reconstruction, remove_small_objects
import utils

from skimage.measure import label
from skimage.color import label2rgb
from skimage.measure import regionprops
import matplotlib.patches as mpatches


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
    bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)

    label_image = label(bone)
    mid = label_image[:, 256]
    labels = set(mid[mid != 0])

    for i in range(0, label_image.shape[0]):
        for j in range(0, label_image.shape[1]):
            if label_image[i, j] in labels:
                bone[i, j] = 0
                label_image[i, j] = 0

    bone = closing(bone, closing_radius)
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

# thresholded_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))

ini = 65
end = 70
mask_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))
for z in range(0, img_original.GetDepth()):
    if ini <= z <= end:
        mask_array[:, :, z] = get_bone_mask(img_original[:, :, z])
        # utils.np_show(r)

# ------------------------ AQUI CONTINUA ------------------------

label_image = label(mask_array)

for z in range(0, img_original.GetDepth()):
    if ini <= z <= end:
        image_label_overlay = label2rgb(label_image[:, :, z], image=mask_array[:, :, z])
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image_label_overlay)

        for region in regionprops(label_image[:, :, z]):
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

plt.show()

# Scikit Image
# for i in range(0, imgOriginal.GetDepth()):
#     tifffile.imsave(outPath+'test-'+'{:03d}'.format(i)+'.tif', boundaries['boundaries_array'][:, :, i])


# DISTRIBUCION DEL RUIDO OBTENIDA CON IMAGEJ: Mean: -1021.905 Std: 44.194
