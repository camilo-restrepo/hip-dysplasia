import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, reconstruction, remove_small_objects, binary_closing, square, opening, watershed
from skimage import feature
from skimage.filters import sobel
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, rank
import utils

from skimage.measure import label
from skimage.color import label2rgb
from skimage.measure import regionprops, find_contours
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
    img_smooth = smooth_filter.Execute(image)
    img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)

    # Segmentacion del cuerpo
    body_threshold = threshold_otsu(img_smooth_array)
    body = img_smooth_array > body_threshold
    body = np.multiply(body, img_smooth_array)

    # Segmentacion de los huesos
    bone_threshold = threshold_otsu(body)
    bone = body > bone_threshold

    bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
    bone = ndi.binary_fill_holes(bone.astype(bool))

    label_image = label(bone)
    mid = label_image[:, 256]
    labels = set(mid[mid != 0])

    for i in range(0, label_image.shape[0]):
        for j in range(0, label_image.shape[1]):
            if label_image[i, j] in labels:
                bone[i, j] = 0
                label_image[i, j] = 0

    # img_array = np.multiply(bone, img_smooth_array)
    # utils.np_show(img_array)
    # utils.show_hist(img_array)

    # edges = feature.canny(img_array, sigma=2.0).astype(int)
    #edges[edges > 0] = 1000
    # r = edges + img_array

    # bone = closing(bone, closing_radius)
    # seed = np.copy(bone)
    # seed[1:-1, 1:-1] = bone.max()
    # mask = bone
    # bone = reconstruction(seed, mask, method='erosion')
    # utils.np_show(bone)
    return bone


def get_segmented_image(image):
    img_array = SimpleITK.GetArrayFromImage(image)
    mask = get_bone_mask(image)
    img_array = np.multiply(img_array, mask)
    # img_array[img_array < 0] = 0
    return img_array


ini = 42
end = 49
mask_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))
for z in range(0, img_original.GetDepth()):
    # if ini <= z <= end:
        mask_array[:, :, z] = get_bone_mask(img_original[:, :, z])


# ------------------------ AQUI CONTINUA ------------------------

def pixel_belongs_to_boundary(img_before, img, img_after, x, y):
    pixel = img[x, y]
    if pixel != 0:
        neighbors = [
            # img_array[x - 1][y - 1][z],
            img[x - 1][y],
            #img_array[x - 1][y + 1][z],
            img[x][y - 1],
            img[x][y + 1],
            #img_array[x + 1][y - 1][z],
            img[x + 1][y],
            #img_array[x + 1][y + 1][z],
            img_before[x][y],
            img_after[x][y]
        ]

        for n in neighbors:
            if n == 0:
                return True
    return False


def compute_boundary(img_before, img, img_after):
    width = img.shape[0]
    height = img.shape[1]
    boundaries_array = np.zeros_like(img)
    e_b_list = []

    for index, x in np.ndenumerate(img):
        if x != 0 and 0 < index[0] < width-1 and 0 < index[1] < height-1:
            if pixel_belongs_to_boundary(img_before, img, img_after, index[0], index[1]):
                boundaries_array[index[0], index[1]] = x
                e_b_list.append(index)

    return {'boundaries_array': boundaries_array, 'e_b': e_b_list}


for z in range(0, img_original.GetDepth()):
    if ini <= z <= end:
        before = mask_array[:, :, z-1]
        image = mask_array[:, :, z-1]
        after = mask_array[:, :, z+1]
        r = compute_boundary(before, image, after)
        l1 = r['e_b'][0]
        im = SimpleITK.GetArrayFromImage(img_original[:, :, z])
        a = l1[0] - 5
        b = l1[0] + 6
        c = l1[1] - 5
        d = l1[1] + 6
        e = im[a:b, c:d]
        utils.np_show(e)
        print e.shape


        #utils.np_show(r['boundaries_array'])


# label_img = label(mask_array)
# for z in range(0, img_original.GetDepth()):
#     if ini <= z <= end:
#         label_img = label(mask_array[:, :, z])
#         image_label_overlay = label2rgb(label_img, image=mask_array[:, :, z])
#         fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#         ax.imshow(image_label_overlay)
#
#         for region in regionprops(label_img):
#             minr, minc, maxr, maxc = region.bbox
#             rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
#             ax.add_patch(rect)

plt.show()

# Scikit Image
# for i in range(0, imgOriginal.GetDepth()):
#     tifffile.imsave(outPath+'test-'+'{:03d}'.format(i)+'.tif', boundaries['boundaries_array'][:, :, i])


# DISTRIBUCION DEL RUIDO OBTENIDA CON IMAGEJ: Mean: -1021.905 Std: 44.194
