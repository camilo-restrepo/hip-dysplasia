import numpy as np
import SimpleITK
import matplotlib.pyplot as plt


def remove_noise(sitk_image):
    smooth_filter = SimpleITK.CurvatureFlowImageFilter()
    smooth_filter.SetTimeStep(0.125)
    smooth_filter.SetNumberOfIterations(5)
    img_smooth = smooth_filter.Execute(sitk_image)
    return SimpleITK.GetArrayFromImage(img_smooth)


def load_dicom(path):
    reader = SimpleITK.ImageSeriesReader()
    filenames_dicom = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(filenames_dicom)
    img_original = reader.Execute()
    img_smooth_array = remove_noise(img_original)
    img_original_array = SimpleITK.GetArrayFromImage(img_original)
    return img_original_array, img_smooth_array


def sitk_show(img):
    img_array = SimpleITK.GetArrayFromImage(img)
    plt.figure()
    plt.imshow(img_array, cmap='Greys_r')


def np_show(img):
    plt.figure()
    plt.imshow(img, cmap='Greys_r', interpolation="nearest")


def histogram_without_background(img):
    hist_filter = SimpleITK.MinimumMaximumImageFilter()
    hist_filter.Execute(img)
    histogram = np.histogram(SimpleITK.GetArrayFromImage(img), bins=np.arange(hist_filter.GetMinimum(),
                                                                              hist_filter.GetMaximum()))
    plt.figure(2)
    histogram[0][0] = 0
    histogram[1][0] = 0
    plt.plot(histogram[1][:-1], histogram[0], lw=1)


def get_stats_without_background(img):
    # img_array = SimpleITK.GetArrayFromImage(img)
    img_array = img[img > 0]
    return {"mean": np.mean(img_array), "std": np.std(img_array), "max": np.max(img_array), "min": np.min(img_array)}


def get_stats(img):
    img_array = SimpleITK.GetArrayFromImage(img)
    return {"mean": np.mean(img_array), "std": np.std(img_array), "max": np.max(img_array), "min": np.min(img_array)}


def show_complete_hist(img):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(np.min(img), np.max(img)), fc='k', ec='k')


def show_positive_hist(img):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(1, np.max(img)), fc='k', ec='k')


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)
