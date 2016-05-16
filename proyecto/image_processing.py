import SimpleITK
import numpy as np
from skimage.morphology import remove_small_objects, closing, disk, erosion, dilation
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from scipy.spatial.distance import euclidean
from skimage.segmentation import clear_border
import region_growing
import matplotlib.pyplot as plt
from utils import Formatter
from sklearn import mixture
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans


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
        self.boundaries = {}
        self.img_boundaries = {}
        self.valleys = {}

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
        self.boundaries = {}
        self.img_boundaries = {}
        self.valleys = {}

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
    # ITERATIVE ADAPTATIVE RECLASSIFICATION
    # --------------------------------------------------------------------------------------------------------------------

    def get_window(self, point, size=5, depth=1):
        z = point[0]
        x = point[1]
        y = point[2]

        image = self.emphasized[self.leg_key]
        x_ini = x - size
        x_end = x + (size + 1)
        y_ini = y - size
        y_end = y + (size + 1)
        z_ini = z - depth
        z_end = z + (depth + 1)
        window = image[z_ini:z_end, x_ini:x_end, y_ini:y_end]
        return window

    def compute_px_model(self, px):
        process = px[3]
        px = (px[0], px[1], px[2])
        if not process:
            g = mixture.GMM(n_components=2)
            window = self.get_window(px)

            g.fit(np.reshape(window, (window.size, 1)))

            cntr = g.means_
            v = 0
            if cntr[0, 0] < cntr[1, 0]:
                v = 1

            predictions = g.predict_proba(np.reshape(window, (window.size, 1)))
            predictions = np.reshape(predictions[:, v], window.shape)

            predictions = predictions[2, :, :]
            predictions[predictions > 0.5] = 1
            predictions[predictions <= 0.5] = 0
            return (px, predictions)
        return (px, None)

    def pixel_belongs_to_boundary(self, x, y, z):
        mask = self.initial_segmented[self.leg_key]
        neighbors = np.array([
            # mask[z, x - 1, y - 1],
            mask[z, x - 1, y],
            # mask[z, x - 1, y + 1],
            mask[z, x, y - 1],
            mask[z, x, y + 1],
            # mask[z, x + 1, y - 1],
            mask[z, x + 1, y],
            # mask[z, x + 1, y + 1],
            mask[z - 1, x, y],
            mask[z + 1, x, y]
        ])
        return np.any(neighbors == 0)

    def compute_boundary(self):
        mask = self.initial_segmented[self.leg_key]
        width = mask.shape[1]
        height = mask.shape[2]
        depth = mask.shape[0]
        e_b = set()
        bound = np.zeros_like(mask)

        white_pxs = np.argwhere(mask == 1)
        print len(white_pxs)
        for r in range(0, white_pxs.shape[0]):
            px = white_pxs[r, :]
            z, x, y = px[0], px[1], px[2]
            if 2 < z < depth - 3 and 11 < y < width - 11 and 11 < x < height - 11:
                # if not has_black_neighbors(z, x, y):
                if self.pixel_belongs_to_boundary(x, y, z):
                    e_b.add((z, x, y))
                    bound[z, x, y] = 1
                    # e_b2.add((z, x, y, (z, x, y) in prediction_cache))
                    # get_window((z, x, y))

        return e_b, bound

    # --------------------------------------------------------------------------------------------------------------------
    # INITIAL SEGMENTATION
    # --------------------------------------------------------------------------------------------------------------------

    def initial_segmentation(self):
        image = self.emphasized[self.leg_key]
        binary_img = np.zeros_like(image)
        for z in range(0, image.shape[0]):
            th = threshold_otsu(image[z, :, :])
            binary_img[z, :, :] = image[z, :, :] > th
            tmp = np.multiply(image[z, :, :], binary_img[z, :, :])
            tmp[tmp < 0] = 0
            th = threshold_otsu(tmp)
            binary_img[z, :, :] = tmp > th
            binary_img[z, :, :] = clear_border(binary_img[z, :, :])
            binary_img[z, :, :] = remove_small_objects(binary_img[z, :, :].astype(bool), 5)

            # if 130 <= z <= image.shape[0]:
            #     fig = plt.figure(z)
            #     a = fig.add_subplot(1, 2, 1)
            #     imgplot = plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
            #     a.format_coord = Formatter(image[z, :, :])
            #     a = fig.add_subplot(1, 2, 2)
            #     imgplot = plt.imshow(binary_img[z, :, :], cmap='Greys_r', interpolation="none")
            #     plt.show()
        return binary_img

    def initial_segmentation_erosion_dilation(self):
        binary_img = self.initial_segmented[self.leg_key].copy()

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
            # if 25 <= z <= 41:
            #     fig = plt.figure(z)
            #     a = fig.add_subplot(1, 2, 1)
            #     imgplot = plt.imshow(binary_img[z, :, :], cmap='Greys_r', interpolation="none")
            #     a = fig.add_subplot(1, 2, 2)
            #     imgplot = plt.imshow(img, cmap='Greys_r', interpolation="none")
            #     plt.show()

            binary_img[z, :, :] = img

        return binary_img

    def get_femur(self):
        segmented_leg = self.initial_segmented[self.leg_key].copy()
        coordinates = []

        selem = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

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
            for region in regionprops(label_image):
                coordinates.append(region.centroid[1])
            promedio = np.mean(coordinates)
            coordinates = [promedio]

            region = region_growing.simple_2d_binary_region_growing(segmented_leg[z, :, :], seed_points, promedio)

            image[z, :, :] = 0
            for px in region:
                image[z, px[1], px[2]] = 1

            for j in range(0, 3):
                image[z, :, :] = dilation(image[z, :, :], disk(2))
            image[z, :, :] = ndi.binary_fill_holes(image[z, :, :])
            for j in range(0, 3):
                image[z, :, :] = erosion(image[z, :, :], disk(2))

            # if 44 <= z <= segmented_leg.shape[0]:
            #     fig = plt.figure(z)
            #     fig.add_subplot(1, 2, 1)
            #     plt.imshow(segmented_leg[z, :, :], cmap='Greys_r', interpolation="none")
            #     fig.add_subplot(1, 2, 2)
            #     plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
            #     plt.show()

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, image.shape[0]):
                image[z, :, :] = np.fliplr(image[z, :, :])

        return image

    def get_hip(self):
        initial_seg = self.initial_segmented[self.leg_key]
        femur = self.segmented_legs[self.leg_key]
        result = initial_seg - femur
        result[result < 0] = 0

        selem = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ])

        for z in range(0, result.shape[0]):
            img = result[z, :, :]
            img = remove_small_objects(img.astype(bool), 20)
            for j in range(0, 3):
                img = dilation(img, selem)
            img = ndi.binary_fill_holes(img)
            for j in range(0, 3):
                img = erosion(img, selem)
            img = remove_small_objects(img.astype(bool), 150)
            # img = clear_border(img)

            # if 44 <= z <= result.shape[0]:
            #     fig = plt.figure(z)
            #     fig.add_subplot(1, 2, 1)
            #     plt.imshow(result[z, :, :], cmap='Greys_r', interpolation="none")
            #     fig.add_subplot(1, 2, 2)
            #     plt.imshow(img, cmap='Greys_r', interpolation="none")
            #     plt.show()

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
            self.no_noise[leg_key] = self.remove_noise_curvature_flow()
            self.emphasized[leg_key] = self.get_valley_emphasized_image()
            self.initial_segmented[leg_key] = self.initial_segmentation()
            self.segmented_legs[leg_key] = self.get_femur()
            self.segmented_hips[leg_key] = self.get_hip()
