import SimpleITK
import numpy as np
from skimage.morphology import remove_small_objects, closing, disk, erosion, dilation
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
import region_growing
import matplotlib.pyplot as plt
from utils import Formatter
from skimage.filters import threshold_otsu


class ImageProcessing:
    def __init__(self):
        self.RIGHT_LEG = 'right_leg'
        self.LEFT_LEG = 'left_leg'
        self.PathDicom = ''
        self.leg_key = ''
        self.spacing = ()

        self.segmented_legs = {}
        self.segmented_hips = {}
        self.legs = {}

    def initialize(self):
        self.PathDicom = ''
        self.leg_key = ''
        self.spacing = ()

        self.segmented_legs = {}
        self.segmented_hips = {}
        self.legs = {}

    # --------------------------------------------------------------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------------------------------------------------------------

    def load_dicom(self):
        reader = SimpleITK.ImageSeriesReader()
        filenames_dicom = reader.GetGDCMSeriesFileNames(self.PathDicom)
        reader.SetFileNames(filenames_dicom)
        img_original = reader.Execute()
        self.spacing = img_original.GetSpacing()
        original_array = SimpleITK.GetArrayFromImage(img_original)
        return original_array

    # --------------------------------------------------------------------------------------------------------------------
    # SEPARATE LEGS
    # --------------------------------------------------------------------------------------------------------------------

    def get_sides_center_coordinates(self, img_original_array, remove_small_objects_size=80):
        bone = np.zeros_like(img_original_array[0, :, :])
        bone[img_original_array[0, :, :] < 200] = 0
        bone[img_original_array[0, :, :] > 200] = 1

        bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
        label_image = label(bone)

        centroids = [region.centroid for region in regionprops(label_image)]
        return centroids

    def get_rows(self, row):
        row_ini = 0
        if row - 128 > 0:
            row_ini = row - 128
        row_end = row_ini + 256
        return {'row_ini': row_ini, 'row_end': row_end}

    def get_legs(self):
        img_original_array = self.load_dicom()
        centroids = self.get_sides_center_coordinates(img_original_array)
        row1 = centroids[0][0].astype(int)
        rows = self.get_rows(row1)
        right_leg = img_original_array[:, rows['row_ini']:rows['row_end'], 0:256]

        row2 = centroids[1][0].astype(int)
        rows = self.get_rows(row2)
        left_leg = img_original_array[:, rows['row_ini']:rows['row_end'], 256:]

        return {self.RIGHT_LEG: right_leg, self.LEFT_LEG: left_leg}

    # --------------------------------------------------------------------------------------------------------------------
    # REMOVE NOISE
    # --------------------------------------------------------------------------------------------------------------------

    def remove_noise_curvature_flow(self):
        sitk_image = SimpleITK.GetImageFromArray(self.legs[self.leg_key])
        smooth_filter = SimpleITK.CurvatureFlowImageFilter()
        smooth_filter.SetTimeStep(0.125)
        smooth_filter.SetNumberOfIterations(5)
        img_smooth = smooth_filter.Execute(sitk_image)
        return SimpleITK.GetArrayFromImage(img_smooth)

    # --------------------------------------------------------------------------------------------------------------------
    # VALLEY COMPUTATION
    # --------------------------------------------------------------------------------------------------------------------

    def get_valley_image(self, image):
        valley_img = np.zeros_like(image)
        for z in range(0, image.shape[0]):
            valley_img[z, :, :] = closing(image[z, :, :], disk(5))
        valley_img -= image

        return valley_img

    def get_valley_emphasized_image(self):
        image = self.remove_noise_curvature_flow()
        valley_image = self.get_valley_image(image)
        valley_image = image - valley_image
        return valley_image

    # --------------------------------------------------------------------------------------------------------------------
    # INITIAL SEGMENTATION
    # --------------------------------------------------------------------------------------------------------------------

    def initial_segmentation(self, emphasized):
        binary_img = np.zeros_like(emphasized)
        for z in range(0, emphasized.shape[0]):
            th = threshold_otsu(emphasized[z, :, :])
            binary_img[z, :, :] = emphasized[z, :, :] > th
            tmp = np.multiply(emphasized[z, :, :], binary_img[z, :, :])
            tmp[tmp < 0] = 0
            th = threshold_otsu(tmp)
            binary_img[z, :, :] = tmp > th
            binary_img[z, :, :] = remove_small_objects(binary_img[z, :, :].astype(bool), 5)

            # print th
            # if 100 <= z <= image.shape[0]:
            #     fig = plt.figure(z)
            #     a = fig.add_subplot(1, 3, 1)
            #     imgplot = plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
            #     a.format_coord = Formatter(image[z, :, :])
            #     a = fig.add_subplot(1, 3, 2)
            #     imgplot = plt.imshow(tmp, cmap='Greys_r', interpolation="none")
            #     a = fig.add_subplot(1, 3, 3)
            #     imgplot = plt.imshow(binary_img[z, :, :], cmap='Greys_r', interpolation="none")
            #     plt.show()
        return binary_img

    def otsu_segmentation_low_threshold(self, emphasized):
        binary_img = np.zeros_like(emphasized)
        for z in range(0, emphasized.shape[0]):
            th = threshold_otsu(emphasized[z, :, :])
            binary_img[z, :, :] = emphasized[z, :, :] > th
            tmp = np.multiply(emphasized[z, :, :], binary_img[z, :, :])
            tmp[tmp < 0] = 0
            th = threshold_otsu(tmp)

            binary_img[z, :, :] = tmp > th - 100
            # binary_img[z, :, :] = clear_border(binary_img[z, :, :])
            binary_img[z, :, :] = remove_small_objects(binary_img[z, :, :].astype(bool), 5)
        return binary_img

    def get_femur(self, initial_segmentation, otsu_low_threshold):
        segmented_leg = initial_segmentation.copy()
        coordinates = []

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, segmented_leg.shape[0]):
                segmented_leg[z, :, :] = np.fliplr(segmented_leg[z, :, :])

        image = segmented_leg.copy()

        for z in range(1, segmented_leg.shape[0]):
            seed_points = set()
            white_pxs = np.argwhere(image[z - 1, :, :] == 1)
            for px in white_pxs:
                seed_points.add((z, px[0], px[1]))

            label_image = label(image[z - 1, :, :])

            coordinates = [region.centroid[1] for region in regionprops(label_image)]
            promedio = np.mean(coordinates)
            coordinates = [promedio]

            region = region_growing.simple_2d_binary_region_growing(segmented_leg[z, :, :], seed_points, promedio)

            image[z, :, :] = 0
            for px in region:
                image[z, px[1], px[2]] = 1

            image[z, :, :] = self.fill_holes(image[z, :, :], disk(2), 3)

            # if 100 <= z <= segmented_leg.shape[0]:
            # if z == 111 or z == 105 or z == 106:
            #     fig = plt.figure(z)
            #     fig.add_subplot(1, 4, 1)
            #     plt.imshow(segmented_leg[z, :, :], cmap='Greys_r', interpolation="none")
            #     fig.add_subplot(1, 4, 2)
            #     plt.imshow(image[z - 1, :, :], cmap='Greys_r', interpolation="none")
            #     fig.add_subplot(1, 4, 3)
            #     plt.imshow(image[z - 1, :, :] + segmented_leg[z, :, :], cmap='Greys_r', interpolation="none")
            #     fig.add_subplot(1, 4, 4)
            #     plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
            #     plt.show()

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, image.shape[0]):
                image[z, :, :] = np.fliplr(image[z, :, :])

        result = np.zeros_like(otsu_low_threshold)
        for z in range(0, otsu_low_threshold.shape[0]):
            seed_points = set()
            white_pxs = np.argwhere(image[z, :, :] == 1)
            for px in white_pxs:
                seed_points.add((z, px[0], px[1]))

            region = region_growing.simple_2d_binary_region_growing2(otsu_low_threshold[z, :, :], seed_points)

            for px in region:
                result[z, px[1], px[2]] = 1

            result[z, :, :] = self.fill_holes(result[z, :, :], disk(2), 1)

            # if 44 <= z <= image.shape[0] and self.leg_key == self.LEFT_LEG:
            # if z == 111 or z == 105 or z == 106:
            #     fig = plt.figure(z)
            #     a = fig.add_subplot(1, 4, 1)
            #     imgplot = plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
            #     a.format_coord = Formatter(image[z, :, :])
            #     a = fig.add_subplot(1, 4, 2)
            #     imgplot = plt.imshow(binary_img[z, :, :], cmap='Greys_r', interpolation="none")
            #     a = fig.add_subplot(1, 4, 3)
            #     imgplot = plt.imshow(image[z, :, :] + binary_img[z, :, :], cmap='Greys_r', interpolation="none")
            #     a = fig.add_subplot(1, 4, 4)
            #     imgplot = plt.imshow(result[z, :, :], cmap='Greys_r', interpolation="none")
            #     plt.show()

        return result

    def get_hip(self, initial_segmentation, otsu_low_threshold):
        femur = self.segmented_legs[self.leg_key]
        last_femur_slice = 0
        removed_elems = False
        for z in range(0, femur.shape[0]):
            if np.any(femur[z, :, :] == 1):
                last_femur_slice = z
            else:
                break

        image = initial_segmentation - femur
        image[image < 0] = 0

        selem = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

        result = np.zeros_like(otsu_low_threshold)
        for z in range(0, otsu_low_threshold.shape[0]):
            seed_points = set()
            white_pxs = np.argwhere(image[z, :, :] == 1)
            for px in white_pxs:
                seed_points.add((z, px[0], px[1]))

            region = region_growing.simple_2d_binary_region_growing2(otsu_low_threshold[z, :, :], seed_points)

            for px in region:
                result[z, px[1], px[2]] = 1

            result[z, :, :] = ndi.binary_fill_holes(result[z, :, :])
            result[z, :, :] = remove_small_objects(result[z, :, :].astype(bool), 50)

            if np.any(result[z, :, :] == 1) and z > last_femur_slice:
                result[z, :, :] = clear_border(result[z, :, :])
                label_image = label(result[z, :, :])
                regions = regionprops(label_image)
                promedio = np.mean([r.area for r in regions])

                for region in regions:
                    if (region.area < promedio) or (region.centroid[1] > 200 and z > last_femur_slice and self.leg_key == self.RIGHT_LEG) or (region.centroid[1] < 50 and z > last_femur_slice and self.leg_key == self.LEFT_LEG):
                        for coord in region.coords:
                            result[z, coord[0], coord[1]] = 0
            elif z > last_femur_slice and np.all(result[z, :, :] == 0) and not removed_elems:
                result[z:, :, :] = 0
                removed_elems = True

            result[z, :, :] = self.fill_holes(result[z, :, :], disk(1), 1)
            result[z, :, :] = remove_small_objects(result[z, :, :].astype(bool), 150)

            # if 100 <= z <= result.shape[0]:
            #     fig = plt.figure(z)
            #     fig.add_subplot(1, 2, 1)
            #     plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
            #     fig.add_subplot(1, 2, 2)
            #     plt.imshow(result[z, :, :], cmap='Greys_r', interpolation="none")
            #     plt.show()

        # label_image = label(result)
        # for region in regionprops(label_image):
        #     print region.area

        return result

# --------------------------------------------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------------------------------------------

    def fill_holes(self, binary_image, selem, iterations):
        image = binary_image.copy()
        for j in range(0, iterations):
            image = dilation(image, selem)
        image = ndi.binary_fill_holes(image)
        for j in range(0, iterations):
            image = erosion(image, selem)
        return image

# --------------------------------------------------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------------------------------------------------

    def execute(self, folder_path):
        self.initialize()
        self.PathDicom = folder_path
        self.legs = self.get_legs()

        for leg_key in self.legs.keys():
            self.leg_key = leg_key
            emphasized = self.get_valley_emphasized_image()
            initial_segmentation = self.initial_segmentation(emphasized)
            otsu_low_threshold = self.otsu_segmentation_low_threshold(emphasized)
            self.segmented_legs[leg_key] = self.get_femur(initial_segmentation, otsu_low_threshold)
            self.segmented_hips[leg_key] = self.get_hip(initial_segmentation, otsu_low_threshold)
