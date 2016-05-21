import SimpleITK
import numpy as np
from skimage.morphology import remove_small_objects, closing, disk, erosion, dilation
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from sklearn import mixture
from scipy import ndimage as ndi
import region_growing
import matplotlib.pyplot as plt
import time


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
        binary_img = np.zeros(emphasized.shape)
        for z in range(0, emphasized.shape[0]):
            th = threshold_otsu(emphasized[z, :, :])
            binary_img[z, :, :] = emphasized[z, :, :] > th
            tmp = np.multiply(emphasized[z, :, :], binary_img[z, :, :])
            tmp[tmp < 0] = 0
            th = threshold_otsu(tmp)
            binary_img[z, :, :] = tmp > th
            binary_img[z, :, :] = remove_small_objects(binary_img[z, :, :].astype(bool), 5)
        binary_img = binary_img.astype('i4')
        return binary_img

    def otsu_segmentation_low_threshold(self, emphasized):
        binary_img = np.zeros(emphasized.shape)
        for z in range(0, emphasized.shape[0]):
            th = threshold_otsu(emphasized[z, :, :])
            binary_img[z, :, :] = emphasized[z, :, :] > th
            tmp = np.multiply(emphasized[z, :, :], binary_img[z, :, :])
            tmp[tmp < 0] = 0
            th = threshold_otsu(tmp)

            binary_img[z, :, :] = tmp > th - 100
            binary_img[z, :, :] = remove_small_objects(binary_img[z, :, :].astype(bool), 5)
        binary_img = binary_img.astype('i4')
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

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, image.shape[0]):
                image[z, :, :] = np.fliplr(image[z, :, :])

        result = np.zeros_like(image)
        for z in range(0, image.shape[0]):
            seed_points = set()
            white_pxs = np.argwhere(image[z, :, :] == 1)
            for px in white_pxs:
                seed_points.add((z, px[0], px[1]))

            region = region_growing.simple_2d_binary_region_growing2(otsu_low_threshold[z, :, :], seed_points)

            for px in region:
                result[z, px[1], px[2]] = 1

            result[z, :, :] = self.fill_holes(result[z, :, :], disk(2), 1)
        result = result.astype('i4')
        return result

    def get_hip(self, femur, initial_segmentation, otsu_low_threshold):
        last_femur_slice = 0
        removed_elems = False
        for z in range(0, femur.shape[0]):
            if np.any(femur[z, :, :] == 1):
                last_femur_slice = z
            else:
                break

        image = initial_segmentation - femur
        image[image < 0] = 0

        result = np.zeros_like(image)
        for z in range(0, image.shape[0]):
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
                    if (region.area < promedio) or (region.centroid[1] > 200 and z > last_femur_slice and self.leg_key == self.RIGHT_LEG) or (
                                region.centroid[1] < 50 and z > last_femur_slice and self.leg_key == self.LEFT_LEG):
                        for coord in region.coords:
                            result[z, coord[0], coord[1]] = 0
            elif z > last_femur_slice and np.all(result[z, :, :] == 0) and not removed_elems:
                result[z:, :, :] = 0
                removed_elems = True

        label_image = label(result)
        for z in range(0, result.shape[0]):
            regions = regionprops(label_image[z, :, :])
            areas = [region.area for region in regions]
            if len(areas) > 0:
                max_area = np.max(areas)
                for region in regions:
                    if region.area != max_area:
                        for coords in region.coords:
                            result[z, coords[0], coords[1]] = 0

            result[z, :, :] = self.fill_holes(result[z, :, :], disk(1), 1)
            result[z, :, :] = remove_small_objects(result[z, :, :].astype(bool), 150)

        result = result.astype('i4')
        return result

    def refine_femur_segmentation(self, otsu_low_threshold, emphasized_img):
        hip = self.segmented_hips[self.leg_key].copy()
        image = otsu_low_threshold - hip
        image[image < 0] = 0

        hip[hip == 1] = 2
        hip[hip == 0] = 1
        hip[hip == 2] = 0
        no_hip = np.multiply(hip, emphasized_img)
        no_hip[no_hip < 0] = 0

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, image.shape[0]):
                image[z, :, :] = np.fliplr(image[z, :, :])
                no_hip[z, :, :] = np.fliplr(no_hip[z, :, :])

        for z in range(0, image.shape[0]):
            image[z, 230:, :] = 0
            image[z, :50, :] = 0
            image[z, :, 200:] = 0

        image = self.clean_by_area(image)

        for z in range(0, image.shape[0]):
            image[z, :, :] = self.fill_holes(image[z, :, :], disk(1), 3)

        label_image = label(image)
        min_row, min_col, max_row, max_col = 0, 0, 0, 0
        max_area = 0
        for z in range(0, image.shape[0]):
            regions = regionprops(label_image[z, :, :])
            for region in regions:
                if region.area > max_area:
                    min_row, min_col, max_row, max_col = region.bbox
                    max_area = region.area

        rect_width = max_col - min_col
        rect_height = max_row - min_row

        image = image.astype('i4')

        for z in range(0, image.shape[0]):
            regions = regionprops(label_image[z, :, :])
            x, y = 0, 0
            for region in regions:
                x, y = region.centroid[0], region.centroid[1]
            min_row1, min_col1 = int(x - (rect_height / 2)), int(y - (rect_width / 2))
            max_row1, max_col1 = int(x + (rect_height / 2)), int(y + (rect_width / 2))

            training = no_hip[z, min_row1:max_row1 + 1, min_col1 - 10:max_col1 + 20]
            if training.size > 0:
                gmm_nohip = mixture.GMM(n_components=4)
                gmm_nohip.fit(np.reshape(training, (-1, 1)))
                means = gmm_nohip.means_.ravel()
                pixeles = no_hip[z, min_row1 - 10:max_row1 + 10, min_col1 - 10:max_col1 + 20]
                predictions = np.reshape(gmm_nohip.predict(np.reshape(pixeles, (-1, 1))), pixeles.shape)

                predictions_img = np.zeros((no_hip.shape[1], no_hip.shape[2]))
                predictions_img[min_row1 - 10:max_row1 + 10, min_col1 - 10:max_col1 + 20] = predictions

                for idx, m in enumerate(means):
                    predictions_img[predictions_img == idx] = m

                sorted_meand = np.sort(means)
                for idx, m in enumerate(sorted_meand):
                    if idx < 2:
                        predictions_img[predictions_img == m] = 0
                    else:
                        predictions_img[predictions_img == m] = 1

                predictions_img[:min_row1 - 10, :] = 0
                predictions_img[max_row1 + 10:, :] = 0
                predictions_img[:, :min_col1 - 10] = 0
                predictions_img[:, max_col1 + 20:] = 0

                predictions_img = predictions_img.astype('i4')
                tmp = predictions_img + image[z, :, :]
                tmp[tmp != 0] = 1
                tmp = ndi.binary_fill_holes(tmp)
                tmp = remove_small_objects(tmp.astype(bool), 100)
                image[z, :, :] = tmp.astype('i4')
                last_z = z
            else:
                break

        image = self.clean_by_area(image)

        if self.leg_key == self.LEFT_LEG:
            for z in range(0, image.shape[0]):
                image[z, :, :] = np.fliplr(image[z, :, :])

        # for z in range(0, last_z):
        #     if z > 110:
        #         fig = plt.figure(z)
        #         a = fig.add_subplot(1, 2, 1)
        #         img = plt.imshow(emphasized_img[z, :, :], cmap='Greys_r', interpolation="none")
        #         a = fig.add_subplot(1, 2, 2)
        #         img = plt.imshow(image[z, :, :], cmap='Greys_r', interpolation="none")
        #         plt.show()

        return image

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

    def clean_by_area(self, binary_image):
        image = binary_image.copy()
        image = ndi.binary_fill_holes(image)

        label_image = label(binary_image)
        initial_label = regionprops(label_image[0, :, :])[0].label

        for z in range(0, image.shape[0]):
            regions = regionprops(label_image[z, :, :])
            for region in regions:
                if region.label != initial_label:
                    for coords in region.coords:
                        image[z, coords[0], coords[1]] = 0

        for z in range(0, image.shape[0]):
            label_image = label(image[z, :, :], connectivity=1)
            regions = regionprops(label_image)
            if len(regions) > 1:
                max_area = np.max([r.area for r in regions])
                for region in regions:
                    if region.centroid[1] > 120 and region.area < max_area:
                        for coords in region.coords:
                            image[z, coords[0], coords[1]] = 0

        return image


# --------------------------------------------------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------------------------------------------------

    def execute(self, folder_path):
        self.initialize()
        self.PathDicom = folder_path
        start = time.time()
        self.legs = self.get_legs()
        stop = time.time()
        print 'legs: ', (stop - start)

        for leg_key in self.legs.keys():
            self.leg_key = leg_key
            start = time.time()
            emphasized = self.get_valley_emphasized_image()
            stop = time.time()
            print 'emphasized: ', (stop - start)
            start = time.time()
            initial_segmentation = self.initial_segmentation(emphasized)
            stop = time.time()
            print 'initial: ', (stop - start)
            start = time.time()
            otsu_low_threshold = self.otsu_segmentation_low_threshold(emphasized)
            stop = time.time()
            print 'otsu: ', (stop - start)
            start = time.time()
            femur = self.get_femur(initial_segmentation, otsu_low_threshold)
            stop = time.time()
            print 'femur: ', (stop - start)
            start = time.time()
            self.segmented_hips[leg_key] = self.get_hip(femur, initial_segmentation, otsu_low_threshold)
            stop = time.time()
            print 'hip: ', (stop - start)
            start = time.time()
            self.segmented_legs[leg_key] = self.refine_femur_segmentation(otsu_low_threshold, emphasized)
            stop = time.time()
            print 'leg: ', (stop - start)
