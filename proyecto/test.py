import numpy as np
import SimpleITK
import matplotlib.pyplot as plt

PathDicom = "D:\imagenes\ALMANZA_RUIZ_JUAN_CARLOS\TAC_DE_PELVIS - 84441\_Bone_30_2"
outPath = "D:\imagenes\out"


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
confidenceConnectedFilter = SimpleITK.ConfidenceConnectedImageFilter()
confidenceConnectedFilter.SetNumberOfIterations(5)
confidenceConnectedFilter.SetMultiplier(1)
confidenceConnectedFilter.SetReplaceValue(1)
overlayFilter = SimpleITK.LabelOverlayImageFilter()
grayscale_fill_filter = SimpleITK.GrayscaleFillholeImageFilter()
morphological_filter = SimpleITK.GrayscaleMorphologicalClosingImageFilter()
# morphological_filter.SetKernelRadius([1, 1])


ini = 37
end = 37
for i in range(0, img_original.GetDepth()):
    if ini <= i <= end:
        imagen = img_original[:, :, i]
        img_smooth = smooth_filter.Execute(imagen)
        lstSeeds = [(100, 307), (171, 362), (222, 266), (306, 274), (342, 351), (371, 307), (435, 316)]
        confidenceConnectedFilter.SetSeedList(lstSeeds)
        a = confidenceConnectedFilter.Execute(img_smooth)
        a = morphological_filter.Execute(a)
        a = grayscale_fill_filter.Execute(a)

        imgT1SmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(img_smooth), a.GetPixelID())
        b = overlayFilter.Execute(imgT1SmoothInt, a)

        sitk_show(a)
        sitk_show(b)

plt.show()
