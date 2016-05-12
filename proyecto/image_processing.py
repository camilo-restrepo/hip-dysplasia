import SimpleITK
import numpy as np
from skimage.morphology import remove_small_objects, closing, disk, erosion, dilation
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from scipy.spatial.distance import euclidean
from skimage.segmentation import clear_border
import region_growing
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.filters import threshold_otsu


class ImageProcessing:
    def __init__(self):
        self.RIGHT_LEG = 'right_leg'
        self.LEFT_LEG = 'left_leg'
        self.PathDicom = ''
        self.leg_key = ''
        self.spacing = ()
        self.no_noise = {}
        self.emphasized = {}
        self.initial_segmented = {}
        self.segmented_legs = {}
        self.segmented_hips = {}
        self.legs = {}
        self.img_original_array = np.empty((512, 512, 90))

    def initialize(self):
        self.PathDicom = ''
        self.leg_key = ''
        self.spacing = ()
        self.no_noise = {}
        self.emphasized = {}
        self.initial_segmented = {}
        self.segmented_legs = {}
        self.segmented_hips = {}
        self.legs = {}
        self.img_original_array = np.empty((512, 512, 90))

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

    def get_sides_center_coordinates(self, remove_small_objects_size=80):
        bone = np.zeros_like(self.img_original_array[0, :, :])
        bone[self.img_original_array[0, :, :] < 200] = 0
        bone[self.img_original_array[0, :, :] > 200] = 1

        bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
        label_image = label(bone)

        centroids = []
        for region in regionprops(label_image):
            centroids.append(region.centroid)
        return centroids

    def get_rows(self, row):
        row_ini = 0
        if row - 128 > 0:
            row_ini = row - 128
        row_end = row_ini + 256
        return {'row_ini': row_ini, 'row_end': row_end}

    def get_legs(self):
        centroids = self.get_sides_center_coordinates()
        row1 = centroids[0][0].astype(int)
        rows = self.get_rows(row1)
        # right_leg = img_smooth_array[:, rows['row_ini']:rows['row_end'], 0:256]
        right_leg = self.img_original_array[:, rows['row_ini']:rows['row_end'], 0:256]

        row2 = centroids[1][0].astype(int)
        rows = self.get_rows(row2)
        # left_leg = img_smooth_array[:, rows['row_ini']:rows['row_end'], 256:]
        left_leg = self.img_original_array[:, rows['row_ini']:rows['row_end'], 256:]

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

    def remove_noise_anisotropic(self):
        sitk_image = SimpleITK.GetImageFromArray(self.legs[self.leg_key])
        sitk_image = SimpleITK.Cast(sitk_image, SimpleITK.sitkFloat64)

        extract_filter = SimpleITK.ExtractImageFilter()
        extract_filter.SetSize([self.legs[self.leg_key].shape[1], self.legs[self.leg_key].shape[2], 0])

        smooth_filter = SimpleITK.GradientAnisotropicDiffusionImageFilter()
        smooth_filter.SetTimeStep(0.06)
        smooth_filter.SetNumberOfIterations(50)
        smooth_filter.SetConductanceParameter(0.5)

        img_smooth = SimpleITK.GetArrayFromImage(smooth_filter.Execute(sitk_image))

        # img_smooth = np.zeros(legs[leg_key].shape)
        # for z in range(0, img_smooth.shape[0]):
        #     extract_filter.SetIndex([0, 0, z])
        #     img_smooth[z, :, :] = SimpleITK.GetArrayFromImage(smooth_filter.Execute(sitk_image[:, :, z]))
        return img_smooth

    # --------------------------------------------------------------------------------------------------------------------
    # VALLEY COMPUTATION
    # --------------------------------------------------------------------------------------------------------------------

    def get_valley_image(self):
        image = self.no_noise[self.leg_key]
        valley_img = np.zeros_like(image)
        for z in range(0, image.shape[0]):
            valley_img[z, :, :] = closing(image[z, :, :], disk(5))
        valley_img = valley_img - image

        return valley_img

    def get_binary_valley_image(self):
        image = self.no_noise[self.leg_key]
        valley_img = np.zeros_like(image)
        for z in range(0, image.shape[0]):
            valley_img[z, :, :] = closing(image[z, :, :], disk(5))
        valley_img = valley_img - image

        tmp = np.zeros(valley_img.shape)
        tmp[valley_img > 60] = 1
        tmp[valley_img < 60] = 0
        for z in range(0, image.shape[0]):
            tmp[z, :, :] = remove_small_objects(tmp[z, :, :].astype(bool), 20)
            tmp[z, :, :] = ndi.binary_fill_holes(tmp[z, :, :])

        return tmp

    def get_valley_emphasized_image(self):
        image = self.no_noise[self.leg_key]
        valley_image = self.get_valley_image()
        valley_image = image - valley_image
        return valley_image

    # --------------------------------------------------------------------------------------------------------------------
    # INITIAL SEGMENTATION
    # --------------------------------------------------------------------------------------------------------------------

    def initial_segmentation(self):
        valley = self.get_binary_valley_image()
        image = self.emphasized[self.leg_key]
        binary_img = np.zeros_like(image)
        # binary_img = image > 200
        for z in range(0, image.shape[0]):
            tmp_result = np.zeros_like(image[z, :, :])
            threshold = threshold_otsu(image[z, :, :])
            binary_img[z, :, :] = image[z, :, :] > threshold
            np.multiply(image[z, :, :], binary_img[z, :, :], tmp_result)
            tmp_result[tmp_result < 0] = 0
            threshold = threshold_otsu(tmp_result)
            binary_img[z, :, :] = tmp_result > threshold
            img = binary_img[z, :, :]
            img = clear_border(img)
            img = ndi.binary_fill_holes(img)
            img = remove_small_objects(img.astype(bool), 50)
            img = ndi.binary_fill_holes(img)
            binary_img[z, :, :] = img

        binary_img = binary_img - valley
        binary_img[binary_img == -1] = 0

        for z in range(0, image.shape[0]):
            img = binary_img[z, :, :]
            img = ndi.binary_fill_holes(img)
            binary_img[z, :, :] = img
        return binary_img

    def initial_segmentation_erosion_dilation(self):
        binary_img = self.initial_segmentation()

        selem = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

        for z in range(0, binary_img.shape[0]):
            img = binary_img[z, :, :]
            for j in range(0, 2):
                img = erosion(img, selem)
            for j in range(0, 2):
                img = dilation(img, selem)
            if 10 <= z <= 20:
                fig = plt.figure(z)
                fig.add_subplot(1, 2, 1)
                plt.imshow(binary_img[z, :, :], cmap='Greys_r', interpolation="none")
                fig.add_subplot(1, 2, 2)
                plt.imshow(img, cmap='Greys_r', interpolation="none")
                plt.show()

            binary_img[z, :, :] = img

        return binary_img

    def femur_separation(self):
        image = self.initial_segmentation_erosion_dilation().copy()
        segmented_leg = self.initial_segmented[self.leg_key].copy()
        # image = segmented_leg.copy()
        init = image.copy()
        coordinates = []
        regions = set()

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, image.shape[0]):
                image[z, :, :] = np.fliplr(image[z, :, :])
                segmented_leg[z, :, :] = np.fliplr(segmented_leg[z, :, :])

        i = 0
        while len(coordinates) == 0:
            label_image = label(image[i, :, :])
            for region in regionprops(label_image):
                coordinates.append(region.centroid[1])
            i += 1

        for z in range(i-1, image.shape[0]):
            label_image = label(image[z, :, :])
            for region in regionprops(label_image):
                centroid = region.centroid[1]
                promedio = np.mean(coordinates)
                distance = euclidean(centroid, promedio)

                if centroid < promedio:
                    coordinates.append(centroid)
                elif distance < 40:
                    coordinates.append(centroid)
                else:
                    for coord in region.coords:
                        label_image[coord[0], coord[1]] = 0
                        image[z, coord[0], coord[1]] = 0

            # image_label_overlay = label2rgb(label_image, image=init[z, :, :])
            # segmented_leg = erosion(segmented_leg, square(1))
            # segmented_leg = dilation(segmented_leg, square(1))
            segmented_slice = segmented_leg[z, :, :]

            if 10 <= z <= 20:
                fig = plt.figure(z)
                fig.add_subplot(1, 4, 1)
                plt.imshow(segmented_slice, cmap='Greys_r', interpolation="none")
                fig.add_subplot(1, 4, 2)
                plt.imshow(init[z, :, :], cmap='Greys_r', interpolation="none")
                fig.add_subplot(1, 4, 3)
                plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
                fig.add_subplot(1, 4, 4)
                plt.imshow(segmented_slice + image[z, :, :], cmap='Greys_r', interpolation="none")
                plt.show()

            seed_points = set()
            for x in range(0, segmented_slice.shape[0]):
                for y in range(0, segmented_slice.shape[1]):
                    if image[z, x, y]:
                        seed_points.add((z, x, y))

            region = region_growing.simple_2d_binary_region_growing(segmented_slice, seed_points, np.mean(coordinates))
            regions |= region

        result = np.zeros_like(image)
        for px in regions:
            result[px[0], px[1], px[2]] = 1

        selem = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

        for z in range(0, result.shape[0]):
            img = result[z, :, :]
            for j in range(0, 3):
                img = dilation(img, selem)
            for j in range(0, 3):
                img = erosion(img, selem)
            img = ndi.binary_fill_holes(img)
            result[z, :, :] = img

        # for z in range(0, result.shape[0]):
        #     if 10 <= z <= 20:
        #         fig = plt.figure(z)
        #         fig.add_subplot(1, 2, 1)
        #         plt.imshow(init[z, :, :], cmap='Greys_r', interpolation="none")
        #         fig.add_subplot(1, 2, 2)
        #         plt.imshow(result[z, :, :], cmap='Greys_r', interpolation="none")
        #         plt.show()

        cut = 0
        for z in range(0, result.shape[0]-3):
            z1 = np.all(result[z + 1, :, :] == 0)
            z2 = np.all(result[z + 2, :, :] == 0)
            if z1 and z2:
                cut = z + 2
                break
        result[cut:, :, :] = 0
        print cut



        if self.leg_key == self.LEFT_LEG:
            for z in range(0, result.shape[0]):
                result[z, :, :] = np.fliplr(result[z, :, :])

        return result

    def get_hip(self):
        initial_seg = self.initial_segmented[self.leg_key]
        femur = self.segmented_legs[self.leg_key]
        result = initial_seg - femur
        result[result <= 0] = 0

        selem = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

        for z in range(0, result.shape[0]):
            img = result[z, :, :]
            img = remove_small_objects(img.astype(bool), 200)
            for j in range(0, 5):
                img = dilation(img, selem)
            for j in range(0, 5):
                img = erosion(img, selem)
            img = ndi.binary_fill_holes(img)
            img = remove_small_objects(img.astype(bool), 80)
            img = clear_border(img)
            result[z, :, :] = img
        return result


# --------------------------------------------------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------------------------------------------------

    def execute(self, folder_path):
        self.initialize()
        self.PathDicom = folder_path
        self.img_original_array = self.load_dicom()
        self.legs = self.get_legs()

        for leg_key in self.legs.keys():
            self.leg_key = leg_key
            if leg_key == self.RIGHT_LEG:
                self.no_noise[leg_key] = self.remove_noise_curvature_flow()
                self.emphasized[leg_key] = self.get_valley_emphasized_image()
                self.initial_segmented[leg_key] = self.initial_segmentation()
                self.segmented_legs[leg_key] = self.femur_separation()
                self.segmented_hips[leg_key] = self.get_hip()
